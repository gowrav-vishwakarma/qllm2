"""Stage B: T2S (TTS via Mimi codec tokens) with both-directions data reuse.

The PAM backbone learns to speak by emitting Mimi codec tokens (delay pattern)
in the unified id-space; the frozen Mimi decoder renders the waveform.

Data reuse (core recipe, plan Stage B): each (audio, text) pair is used as
separate samples in BOTH directions -- S2T (`--task both`) and T2S -- so the
scarce Gujarati data trains understanding and speaking at once. Clean
single-speaker corpora should be added via `--extra_clean` for output voice
quality. The single-sequence round-trip autoencoder form (`--task roundtrip`)
is an OPTIONAL ablation, not the core recipe (copy-from-state shortcut risk);
try it only after S2T and T2S each pass their gates.

Sample layouts (unified vocab):
    t2s        : <lang> text... <tts> [codec delay stream] <eos>
    s2t        : <env> [audio_pad x Tf] <transcribe> <lang> text... <eos>
    roundtrip  : <env> [audio_pad x Tf] <transcribe> <lang> text... <tts> [codec] <eos>

Gate: round-trip WER (Mimi-decode generated speech -> Whisper ASR -> vs text).
Cheap tracked metric: teacher-forced codec next-token accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from v11.duplex.audio_data import load_asr_rows
from v11.duplex.codec import MimiCodec, delay_flatten, delay_unflatten
from v11.duplex.config import get_duplex_config
from v11.duplex.encoder import FrozenWhisperEncoder
from v11.duplex.model import V11DuplexLM
from v11.duplex.tokenizer import (
    AUDIO_PAD, ENV_MARK, EOS, LANG_TOKEN, TRANSCRIBE, TTS, DuplexTokenizer,
)
from v11.duplex.train_asr import build_asr_samples, char_error_rate, collate_asr


# ── Sample builders ──────────────────────────────────────────────────────────

def build_tts_samples(
    rows: List[Dict],
    codec: MimiCodec,
    tokenizer: DuplexTokenizer,
    feat_dim: int,
    max_codec_frames: int = 250,
    max_text_tokens: int = 200,
) -> List[Dict]:
    """text -> codec delay stream. No audio injected (text-only input)."""
    vocab = tokenizer.vocab
    empty_frames = torch.zeros(0, feat_dim, dtype=torch.float32)
    samples: List[Dict] = []
    for r in rows:
        audio = r['audio']
        wf = torch.tensor(audio['array'], dtype=torch.float32)
        codes = codec.encode(wf, int(audio['sampling_rate']))
        if codes.shape[1] > max_codec_frames:
            codes = codes[:, :max_codec_frames]
        stream = delay_flatten(codes, vocab)
        lang_id = LANG_TOKEN.get(str(r.get('lang', 'en')).lower(), LANG_TOKEN['en'])
        text_ids = tokenizer.encode_text(r['text'])[:max_text_tokens]
        if not text_ids or not stream:
            continue

        input_ids = [lang_id] + text_ids + [TTS] + stream + [EOS]
        labels = [-100] * (1 + len(text_ids) + 1) + stream + [EOS]
        samples.append({
            'input_ids': input_ids, 'labels': labels, 'audio_positions': [],
            'frames': empty_frames, 'lang': r.get('lang', 'en'), 'text': r['text'],
            'mode': 't2s',
        })
    return samples


def build_roundtrip_samples(
    rows: List[Dict],
    encoder: FrozenWhisperEncoder,
    codec: MimiCodec,
    tokenizer: DuplexTokenizer,
    max_audio_frames: int = 200,
    stride: int = 4,
    max_codec_frames: int = 250,
    max_text_tokens: int = 200,
) -> List[Dict]:
    """[audio] <transcribe> text <tts> codec-of-same-audio (ablation only)."""
    vocab = tokenizer.vocab
    samples: List[Dict] = []
    for r in rows:
        audio = r['audio']
        wf = torch.tensor(audio['array'], dtype=torch.float32)
        sr = int(audio['sampling_rate'])
        frames = encoder.encode_frames(wf, sr, stride=stride,
                                       max_frames=max_audio_frames).detach().cpu().float()
        T = frames.shape[0]
        codes = codec.encode(wf, sr)
        if codes.shape[1] > max_codec_frames:
            codes = codes[:, :max_codec_frames]
        stream = delay_flatten(codes, vocab)
        lang_id = LANG_TOKEN.get(str(r.get('lang', 'en')).lower(), LANG_TOKEN['en'])
        text_ids = tokenizer.encode_text(r['text'])[:max_text_tokens]
        if not text_ids or not stream:
            continue

        input_ids = [ENV_MARK] + [AUDIO_PAD] * T + [TRANSCRIBE, lang_id]
        labels = [-100] * len(input_ids)
        positions = list(range(1, 1 + T))
        for t in text_ids:  # train text (S2T half of round-trip)
            input_ids.append(t)
            labels.append(t)
        input_ids.append(TTS)
        labels.append(-100)
        for c in stream:    # train codec (T2S half)
            input_ids.append(c)
            labels.append(c)
        input_ids.append(EOS)
        labels.append(EOS)
        samples.append({
            'input_ids': input_ids, 'labels': labels, 'audio_positions': positions,
            'frames': frames, 'lang': r.get('lang', 'en'), 'text': r['text'],
            'mode': 'roundtrip',
        })
    return samples


def build_samples(task: str, rows, encoder, codec, tokenizer, feat_dim, **kw) -> List[Dict]:
    task = task.lower()
    if task == 't2s':
        return build_tts_samples(rows, codec, tokenizer, feat_dim,
                                 max_codec_frames=kw.get('max_codec_frames', 250))
    if task == 'both':
        s2t = build_asr_samples(rows, encoder, tokenizer,
                                max_audio_frames=kw.get('max_audio_frames', 200),
                                stride=kw.get('stride', 4))
        for s in s2t:
            s['mode'] = 's2t'
        t2s = build_tts_samples(rows, codec, tokenizer, feat_dim,
                                max_codec_frames=kw.get('max_codec_frames', 250))
        return s2t + t2s
    if task == 'roundtrip':
        return build_roundtrip_samples(rows, encoder, codec, tokenizer,
                                       max_audio_frames=kw.get('max_audio_frames', 200),
                                       stride=kw.get('stride', 4),
                                       max_codec_frames=kw.get('max_codec_frames', 250))
    raise ValueError(f'Unknown task {task!r}. Use t2s | both | roundtrip.')


class TTSDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Codec generation (masked to codec id-space) ──────────────────────────────

@torch.no_grad()
def generate_speech(
    model: V11DuplexLM,
    text: str,
    lang: str,
    tokenizer: DuplexTokenizer,
    device: torch.device,
    max_codec_tokens: int = 400,
) -> torch.Tensor:
    """Greedy text->codec generation, constrained to codec ids (+eos). Returns codes [K,T]."""
    model.eval()
    vocab = tokenizer.vocab
    lang_id = LANG_TOKEN.get(str(lang).lower(), LANG_TOKEN['en'])
    text_ids = tokenizer.encode_text(text)
    prompt = [lang_id] + text_ids + [TTS]
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    logits, states, _ = model(ids)
    step = ids.shape[1]

    codec_lo, codec_hi = vocab.codec_offset, vocab.total_size
    allow = torch.full((vocab.total_size,), float('-inf'), device=device)
    allow[codec_lo:codec_hi] = 0.0
    allow[EOS] = 0.0

    stream: List[int] = []
    nxt_logits = logits[:, -1] + allow
    nxt = nxt_logits.argmax(-1, keepdim=True)
    for _ in range(max_codec_tokens):
        tid = int(nxt)
        if tid == EOS:
            break
        stream.append(tid)
        logits, states, _ = model(nxt, states=states, step_offset=step)
        step += 1
        nxt = (logits[:, -1] + allow).argmax(-1, keepdim=True)
    if not stream:
        return torch.zeros(vocab.n_codebooks, 0, dtype=torch.long)
    return delay_unflatten(stream, vocab)


@torch.no_grad()
def codec_token_accuracy(model, loader, device) -> Dict:
    """Teacher-forced next-token accuracy on codec label positions (cheap gate)."""
    model.eval()
    total_loss, nb = 0.0, 0
    correct, total = 0, 0
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
        correct += (pred[mask] == tgt[mask]).sum().item()
        total += int(mask.sum().item())
    return {'loss': total_loss / max(1, nb), 'token_acc': correct / max(1, total)}


@torch.no_grad()
def round_trip_wer(model, samples, tokenizer, codec, device, n: int = 8,
                   asr_model_name: str = 'openai/whisper-small') -> Optional[float]:
    """Decode generated speech through Mimi then Whisper ASR; CER vs reference text.

    Best-effort gate: needs the Whisper *decoder* (generation). Returns None if
    unavailable so training never crashes on the metric.
    """
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        proc = WhisperProcessor.from_pretrained(asr_model_name)
        asr = WhisperForConditionalGeneration.from_pretrained(asr_model_name).to(device).eval()
    except Exception as e:  # noqa: BLE001
        print(f'  round_trip_wer skipped ({type(e).__name__}: {e})')
        return None
    import numpy as np
    cers = []
    for s in samples[:n]:
        codes = generate_speech(model, s['text'], s['lang'], tokenizer, device)
        if codes.shape[1] == 0:
            cers.append(1.0)
            continue
        wav = codec.decode(codes)  # 24 kHz
        feats = proc(np.asarray(wav), sampling_rate=24000, return_tensors='pt')
        gen = asr.generate(feats.input_features.to(device), max_new_tokens=128)
        hyp = proc.batch_decode(gen, skip_special_tokens=True)[0]
        cers.append(char_error_rate(s['text'], hyp))
    return sum(cers) / max(1, len(cers))


def save_checkpoint(path: Path, model, opt, cfg, epoch, batch_idx, global_step,
                    history, tokenizer_dir, task, metrics=None):
    tmp = path.with_name(path.name + '.tmp')
    ckpt = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'config': cfg,
            'epoch': epoch, 'batch_idx': batch_idx, 'global_step': global_step,
            'history': history, 'tokenizer_dir': tokenizer_dir, 'task': task}
    if metrics is not None:
        ckpt['metrics'] = metrics
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description='V11 duplex Stage B (TTS / T2S)')
    p.add_argument('--preset', default='duplex_100m')
    p.add_argument('--tokenizer_dir', default='checkpoints_v11_duplex_tokenizer')
    p.add_argument('--task', default='both', choices=['t2s', 'both', 'roundtrip'])
    p.add_argument('--languages', default='hindi,gujarati')
    p.add_argument('--n_per_lang', type=int, default=2000)
    p.add_argument('--n_english', type=int, default=2000)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--max_audio_frames', type=int, default=200)
    p.add_argument('--max_codec_frames', type=int, default=250)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--val_frac', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--whisper', default='openai/whisper-small')
    p.add_argument('--mimi', default='kyutai/mimi')
    p.add_argument('--ckpt_dir', default='')
    p.add_argument('--resume', default='')
    p.add_argument('--init_from', default='', help='warm-start model weights (e.g. Stage A best_model.pt)')
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--save_every_steps', type=int, default=500)
    p.add_argument('--eval_round_trip', action='store_true')
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
    ckpt_dir = Path(args.ckpt_dir or f'checkpoints_v11_{args.preset}_tts')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    encoder = FrozenWhisperEncoder(args.whisper, device=str(device))
    codec = MimiCodec(tokenizer.vocab, model_name=args.mimi, device=str(device))
    model = V11DuplexLM(cfg, audio_feat_dim=encoder.out_dim).to(device)
    print(f'Preset {args.preset}: {model.count_parameters()} | task={args.task}')

    if args.init_from and Path(args.init_from).exists():
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'], strict=False)
        print(f'Warm-started from {args.init_from}')

    langs = [s.strip() for s in args.languages.split(',') if s.strip()]
    rows = load_asr_rows(languages=langs, n_per_lang=args.n_per_lang,
                         include_english=args.n_english > 0, n_english=args.n_english,
                         seed=args.seed)
    print(f'Building {args.task} samples (Whisper frames + Mimi codes)...')
    samples = build_samples(args.task, rows, encoder, codec, tokenizer, encoder.out_dim,
                            max_audio_frames=args.max_audio_frames, stride=args.stride,
                            max_codec_frames=args.max_codec_frames)
    print(f'Built {len(samples)} samples')

    ds = TTSDataset(samples)
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
    resume_path = args.resume or os.environ.get('RESUME', '')
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

    best_acc = 0.0
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
                                batch_idx, global_step, history, args.tokenizer_dir, args.task)
                print(f'  [checkpoint @ step {global_step}]')
            if stop['flag']:
                save_checkpoint(ckpt_dir / 'latest.pt', model, opt, cfg, epoch,
                                batch_idx, global_step, history, args.tokenizer_dir, args.task)
                print(f'Shutdown: saved latest.pt @ step {global_step}')
                return
        start_batch = 0
        val = codec_token_accuracy(model, val_loader, device)
        row = {'epoch': epoch, 'train_loss': epoch_loss / max(1, nb), **val}
        if args.eval_round_trip:
            wer = round_trip_wer(model, val_ds.dataset.samples, tokenizer, codec, device)
            if wer is not None:
                row['round_trip_cer'] = wer
        history.append(row)
        print(f'=== epoch {epoch} train_loss={row["train_loss"]:.4f} '
              f'val_loss={val["loss"]:.4f} codec_acc={val["token_acc"]:.3f} '
              f'{"rtCER=%.3f" % row["round_trip_cer"] if "round_trip_cer" in row else ""} ===')
        save_checkpoint(ckpt_dir / 'latest.pt', model, opt, cfg, epoch,
                        len(train_loader) - 1, global_step, history, args.tokenizer_dir,
                        args.task, metrics=row)
        if val['token_acc'] >= best_acc:
            best_acc = val['token_acc']
            torch.save({'model': model.state_dict(), 'config': cfg, 'metrics': row,
                        'tokenizer_dir': args.tokenizer_dir, 'task': args.task},
                       ckpt_dir / 'best_model.pt')
            print(f'  New best codec_acc={best_acc:.3f} -> best_model.pt')

    elapsed = time.time() - t0
    with open(ckpt_dir / 'metrics.json', 'w') as f:
        json.dump({'stage': 'B_tts', 'task': args.task, 'preset': args.preset,
                   'languages': args.languages, 'elapsed_s': elapsed,
                   'history': history, 'best_codec_acc': best_acc}, f, indent=2)
    print(f'Done in {elapsed:.1f}s. best codec_acc={best_acc:.3f}. Ckpts in {ckpt_dir}')


if __name__ == '__main__':
    main()
