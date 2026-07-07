"""Stage C: unified duplex interface model (joint S2T + T2S + control).

Interleaves the three abilities the earlier stages learned separately into one
conversational sequence, initialized from the Stage A (ASR) + Stage B (TTS)
checkpoints (same backbone, same vocab):

    <listen> <env> [audio_pad x Tu] <transcribe> <lang> user_text
    <speak>  <llm> <lang> reply_text <tts> [codec delay stream] <eos>

Loss is on text, codec, and control positions; audio-in positions are masked.
Barge-in is trained by truncating the codec stream and flipping control back to
<listen> mid-reply (same scenario the POC trained in v11/duplex/interleave.py).

Reply/target sourcing:
  --reply_source corpus (default): assistant turn is a REAL second corpus
      utterance — its transcript is the reply text and its audio gives real Mimi
      codec targets. Best teacher, no external TTS dependency.
  --reply_source brain: reply text comes from a brain conversation jsonl
      (v11/duplex/brain.py). Without a teacher-TTS for that text, the <tts>
      codec segment is dropped and T2S is carried by the Stage B init; supply
      real reply audio via the conversation rows to train codec too.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from v11.duplex.audio_data import load_asr_rows
from v11.duplex.codec import MimiCodec, delay_flatten
from v11.duplex.config import get_duplex_config
from v11.duplex.encoder import FrozenWhisperEncoder
from v11.duplex.model import V11DuplexLM
from v11.duplex.tokenizer import (
    AUDIO_PAD, BACKCHANNEL, ENV_MARK, EOS, LANG_TOKEN, LISTEN, LLM, SPEAK,
    TRANSCRIBE, TTS, THINKING_IDS, DuplexTokenizer,
)
from v11.duplex.train_asr import char_error_rate, collate_asr


# ── Pairing ──────────────────────────────────────────────────────────────────

def pair_rows_by_lang(rows: List[Dict], seed: int = 42) -> List[Tuple[Dict, Dict, str]]:
    """Pair consecutive same-language utterances into (user, assistant, lang)."""
    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_lang[str(r.get('lang', 'en')).lower()].append(r)
    rng = random.Random(seed)
    pairs: List[Tuple[Dict, Dict, str]] = []
    for lang, items in by_lang.items():
        rng.shuffle(items)
        for i in range(0, len(items) - 1, 2):
            pairs.append((items[i], items[i + 1], lang))
    rng.shuffle(pairs)
    return pairs


# ── Sample builder ───────────────────────────────────────────────────────────

def build_conversation_samples(
    pairs: List[Tuple[Dict, Dict, str]],
    encoder: FrozenWhisperEncoder,
    codec: MimiCodec,
    tokenizer: DuplexTokenizer,
    barge_in_prob: float = 0.25,
    max_audio_frames: int = 200,
    stride: int = 4,
    max_codec_frames: int = 250,
    max_text_tokens: int = 120,
    reply_texts: Optional[Dict[str, str]] = None,
    seed: int = 42,
) -> List[Dict]:
    vocab = tokenizer.vocab
    rng = random.Random(seed)
    samples: List[Dict] = []
    for user, asst, lang in pairs:
        lang_id = LANG_TOKEN.get(lang, LANG_TOKEN['en'])

        u_wf = torch.tensor(user['audio']['array'], dtype=torch.float32)
        u_sr = int(user['audio']['sampling_rate'])
        frames = encoder.encode_frames(u_wf, u_sr, stride=stride,
                                       max_frames=max_audio_frames).detach().cpu().float()
        Tu = frames.shape[0]
        user_text_ids = tokenizer.encode_text(user['text'])[:max_text_tokens]

        reply_text = (reply_texts or {}).get(user['text'], asst['text'])
        reply_ids = tokenizer.encode_text(reply_text)[:max_text_tokens]

        a_wf = torch.tensor(asst['audio']['array'], dtype=torch.float32)
        codes = codec.encode(a_wf, int(asst['audio']['sampling_rate']))
        if codes.shape[1] > max_codec_frames:
            codes = codes[:, :max_codec_frames]
        codec_stream = delay_flatten(codes, vocab)
        if not user_text_ids or not reply_ids or not codec_stream:
            continue

        input_ids: List[int] = []
        labels: List[int] = []

        def emit(tok: int, train: bool):
            input_ids.append(tok)
            labels.append(tok if train else -100)

        # ── user turn: listen + audio + transcript (S2T) ──
        emit(LISTEN, True)
        emit(ENV_MARK, False)
        audio_positions = list(range(len(input_ids), len(input_ids) + Tu))
        for _ in range(Tu):
            emit(AUDIO_PAD, False)
        emit(TRANSCRIBE, False)
        emit(lang_id, False)
        for t in user_text_ids:
            emit(t, True)

        # ── assistant turn: speak + reply text + codec (T2S) ──
        emit(SPEAK, True)
        emit(LLM, False)
        emit(lang_id, False)
        for t in reply_ids:
            emit(t, True)
        emit(TTS, False)

        do_barge = rng.random() < barge_in_prob
        keep = len(codec_stream)
        if do_barge:
            keep = max(vocab.n_codebooks, (len(codec_stream) // 2))
        for c in codec_stream[:keep]:
            emit(c, True)

        if do_barge:
            emit(BACKCHANNEL if rng.random() < 0.5 else LISTEN, True)
        emit(EOS, True)

        samples.append({
            'input_ids': input_ids, 'labels': labels, 'audio_positions': audio_positions,
            'frames': frames, 'lang': lang, 'text': user['text'], 'reply': reply_text,
            'barge_in': do_barge,
        })
    return samples


class DuplexV2Dataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Eval: bucketed next-token accuracy by id region ──────────────────────────

@torch.no_grad()
def evaluate(model, loader, tokenizer, device) -> Dict:
    model.eval()
    vocab = tokenizer.vocab
    thinking = set(THINKING_IDS)
    total_loss, nb = 0.0, 0
    buckets = {'control': [0, 0], 'text': [0, 0], 'codec': [0, 0]}
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        positions = batch['audio_positions'].to(device)
        frames = batch['frames']
        embeds = model.project_audio(frames.to(device)) if frames.shape[1] > 0 else None
        logits, _, _ = model(input_ids, audio_embeds=embeds, audio_positions=positions)
        total_loss += V11DuplexLM.compute_loss(logits, labels).item()
        nb += 1
        pred = logits[:, :-1].argmax(-1)
        tgt = labels[:, 1:]
        mask = tgt != -100
        pt, tt = pred[mask].tolist(), tgt[mask].tolist()
        for p, t in zip(pt, tt):
            if t in thinking:
                key = 'control'
            elif vocab.is_codec(t):
                key = 'codec'
            elif vocab.is_text(t):
                key = 'text'
            else:
                continue
            buckets[key][1] += 1
            buckets[key][0] += int(p == t)
    out = {'loss': total_loss / max(1, nb)}
    for k, (c, n) in buckets.items():
        out[f'{k}_acc'] = c / max(1, n)
        out[f'{k}_n'] = n
    return out


def save_checkpoint(path: Path, model, opt, cfg, epoch, batch_idx, global_step,
                    history, tokenizer_dir, metrics=None):
    tmp = path.with_name(path.name + '.tmp')
    ckpt = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'config': cfg,
            'epoch': epoch, 'batch_idx': batch_idx, 'global_step': global_step,
            'history': history, 'tokenizer_dir': tokenizer_dir}
    if metrics is not None:
        ckpt['metrics'] = metrics
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _load_weights(model, path, device):
    if path and Path(path).exists():
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'], strict=False)
        print(f'  loaded weights from {path}')
        return True
    return False


def parse_args():
    p = argparse.ArgumentParser(description='V11 duplex Stage C (unified interface)')
    p.add_argument('--preset', default='duplex_100m')
    p.add_argument('--tokenizer_dir', default='checkpoints_v11_duplex_tokenizer')
    p.add_argument('--reply_source', default='corpus', choices=['corpus', 'brain'])
    p.add_argument('--conversations', default='', help='brain jsonl (reply_source=brain)')
    p.add_argument('--languages', default='hindi,gujarati')
    p.add_argument('--n_per_lang', type=int, default=2000)
    p.add_argument('--n_english', type=int, default=2000)
    p.add_argument('--barge_in_prob', type=float, default=0.25)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--max_audio_frames', type=int, default=200)
    p.add_argument('--max_codec_frames', type=int, default=250)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--val_frac', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--whisper', default='openai/whisper-small')
    p.add_argument('--mimi', default='kyutai/mimi')
    p.add_argument('--init_asr', default='', help='Stage A best_model.pt')
    p.add_argument('--init_tts', default='', help='Stage B best_model.pt (applied after ASR)')
    p.add_argument('--ckpt_dir', default='')
    p.add_argument('--resume', default='')
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--save_every_steps', type=int, default=500)
    p.add_argument('--amp', default='bf16', choices=['none', 'bf16', 'fp16'])
    return p.parse_args()


def _amp_ctx(device, amp):
    from contextlib import nullcontext
    if amp == 'none' or device.type != 'cuda':
        return nullcontext()
    dt = torch.bfloat16 if amp == 'bf16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dt)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = DuplexTokenizer.load(args.tokenizer_dir)
    cfg = get_duplex_config(args.preset, vocab_size=tokenizer.vocab.total_size)
    ckpt_dir = Path(args.ckpt_dir or f'checkpoints_v11_{args.preset}_duplex_v2')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    encoder = FrozenWhisperEncoder(args.whisper, device=str(device))
    codec = MimiCodec(tokenizer.vocab, model_name=args.mimi, device=str(device))
    model = V11DuplexLM(cfg, audio_feat_dim=encoder.out_dim).to(device)
    print(f'Preset {args.preset}: {model.count_parameters()}')

    resume_path = args.resume or os.environ.get('RESUME', '')
    if not (resume_path and Path(resume_path).exists()):
        _load_weights(model, args.init_asr, device)  # A first
        _load_weights(model, args.init_tts, device)  # then B (same backbone/vocab)

    reply_texts = None
    if args.reply_source == 'brain':
        if not args.conversations:
            raise ValueError('reply_source=brain requires --conversations jsonl')
        from v11.duplex.brain import load_conversation_dataset
        reply_texts = {r['user']: r['reply'] for r in load_conversation_dataset(args.conversations)}
        print(f'Loaded {len(reply_texts)} brain replies')

    langs = [s.strip() for s in args.languages.split(',') if s.strip()]
    rows = load_asr_rows(languages=langs, n_per_lang=args.n_per_lang,
                         include_english=args.n_english > 0, n_english=args.n_english,
                         seed=args.seed)
    pairs = pair_rows_by_lang(rows, seed=args.seed)
    print(f'Building {len(pairs)} conversation samples...')
    samples = build_conversation_samples(
        pairs, encoder, codec, tokenizer, barge_in_prob=args.barge_in_prob,
        max_audio_frames=args.max_audio_frames, stride=args.stride,
        max_codec_frames=args.max_codec_frames, reply_texts=reply_texts, seed=args.seed)
    print(f'Built {len(samples)} samples '
          f'({sum(s["barge_in"] for s in samples)} barge-in)')

    ds = DuplexV2Dataset(samples)
    n_val = max(1, int(len(ds) * args.val_frac))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    feat_dim = encoder.out_dim
    collate = lambda b: collate_asr(b, feat_dim)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)

    def lr_at(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    start_epoch, start_batch, global_step, history = 1, 0, 0, []
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        if 'optimizer' in ckpt:
            opt.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 1)
        start_batch = ckpt.get('batch_idx', -1) + 1
        global_step = ckpt.get('global_step', 0)
        history = ckpt.get('history', [])
        print(f'Resumed {resume_path}: ep{start_epoch} step{global_step}')

    stop = {'flag': False}

    def _sig(signum, _f):
        stop['flag'] = True
        print(f'\nSignal {signum}: saving latest.pt...', flush=True)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    best = 0.0
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss, nb = 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            for g in opt.param_groups:
                g['lr'] = args.lr * lr_at(global_step)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            positions = batch['audio_positions'].to(device)
            frames = batch['frames']
            with torch.no_grad():
                embeds = model.project_audio(frames.to(device)) if frames.shape[1] > 0 else None
            opt.zero_grad(set_to_none=True)
            with _amp_ctx(device, args.amp):
                logits, _, _ = model(input_ids, audio_embeds=embeds, audio_positions=positions)
                loss = V11DuplexLM.compute_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            nb += 1
            global_step += 1
            if batch_idx % args.log_every == 0:
                print(f'ep{epoch} step{global_step} loss={loss.item():.4f} '
                      f'lr={opt.param_groups[0]["lr"]:.2e}')
            if args.save_every_steps and global_step % args.save_every_steps == 0:
                save_checkpoint(ckpt_dir / 'latest.pt', model, opt, cfg, epoch,
                                batch_idx, global_step, history, args.tokenizer_dir)
                print(f'  [checkpoint @ step {global_step}]')
            if stop['flag']:
                save_checkpoint(ckpt_dir / 'latest.pt', model, opt, cfg, epoch,
                                batch_idx, global_step, history, args.tokenizer_dir)
                print(f'Shutdown: saved latest.pt @ step {global_step}')
                return
        start_batch = 0
        val = evaluate(model, val_loader, tokenizer, device)
        row = {'epoch': epoch, 'train_loss': epoch_loss / max(1, nb), **val}
        history.append(row)
        print(f'=== epoch {epoch} train_loss={row["train_loss"]:.4f} val_loss={val["loss"]:.4f} '
              f'ctrl={val["control_acc"]:.3f} text={val["text_acc"]:.3f} '
              f'codec={val["codec_acc"]:.3f} ===')
        save_checkpoint(ckpt_dir / 'latest.pt', model, opt, cfg, epoch,
                        len(train_loader) - 1, global_step, history, args.tokenizer_dir,
                        metrics=row)
        score = (val['control_acc'] + val['text_acc'] + val['codec_acc']) / 3
        if score >= best:
            best = score
            torch.save({'model': model.state_dict(), 'config': cfg, 'metrics': row,
                        'tokenizer_dir': args.tokenizer_dir}, ckpt_dir / 'best_model.pt')
            print(f'  New best mean_acc={best:.3f} -> best_model.pt')

    elapsed = time.time() - t0
    with open(ckpt_dir / 'metrics.json', 'w') as f:
        json.dump({'stage': 'C_duplex_v2', 'preset': args.preset,
                   'reply_source': args.reply_source, 'languages': args.languages,
                   'elapsed_s': elapsed, 'history': history, 'best_mean_acc': best}, f, indent=2)
    print(f'Done in {elapsed:.1f}s. best mean_acc={best:.3f}. Ckpts in {ckpt_dir}')


if __name__ == '__main__':
    main()
