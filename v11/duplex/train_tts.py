"""Stage B: T2S (TTS via Mimi codec tokens).

Primary recipe (`--task t2s`): text -> Mimi delay-pattern codec stream.
The PAM (or matched transformer) backbone emits codec ids in the unified
vocab; the frozen Mimi decoder renders the waveform. No Whisper / no ASR
objective — this is the focused "is QLLM good at TTS?" run.

    t2s        : <lang> text... <tts> [codec delay stream] <eos>
    s2t / both / roundtrip : kept for the older duplex curriculum

Gate: teacher-forced codec next-token accuracy (cheap, every epoch).
Optional: round-trip CER (Mimi-decode -> Whisper ASR) and saved wavs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from v11.duplex.audio_data import load_tts_rows
from v11.duplex.codec import MimiCodec, delay_flatten, delay_unflatten
from v11.duplex.config import get_duplex_config
from v11.duplex.logutil import elapsed_since, log
from v11.duplex.model import V11DuplexLM
from v11.duplex.tokenizer import EOS, LANG_TOKEN, TTS, DuplexTokenizer
from v11.duplex.train_asr import char_error_rate


LISTEN_PROMPTS: List[Tuple[str, str]] = [
    ('en', 'Hello, this is a test of the voice model.'),
    ('hindi', 'नमस्ते, आप कैसे हैं?'),
    ('gujarati', 'કેમ છો? આ એક પરીક્ષણ છે.'),
]


# ── Sample builders ──────────────────────────────────────────────────────────

def build_tts_samples(
    rows: List[Dict],
    codec: MimiCodec,
    tokenizer: DuplexTokenizer,
    max_codec_frames: int = 250,
    max_text_tokens: int = 200,
    codec_cache: str = '',
    min_codec_frames: int = 4,
) -> List[Dict]:
    """text -> codec delay stream. No audio injected (text-only input)."""
    vocab = tokenizer.vocab
    samples: List[Dict] = []
    skipped = 0
    for i, r in enumerate(rows):
        if i % 50 == 0:
            log(f'  mimi encode {i}/{len(rows)} (kept {len(samples)})')
        audio = r['audio']
        wf = torch.tensor(audio['array'], dtype=torch.float32)
        codes = codec.encode_cached(
            wf, int(audio['sampling_rate']),
            cache_dir=codec_cache or None,
            extra_key=str(r.get('text', ''))[:80],
        )
        if codes.shape[1] < min_codec_frames:
            skipped += 1
            continue
        if codes.shape[1] > max_codec_frames:
            codes = codes[:, :max_codec_frames]
        stream = delay_flatten(codes, vocab)
        lang_id = LANG_TOKEN.get(str(r.get('lang', 'en')).lower(), LANG_TOKEN['en'])
        text_ids = tokenizer.encode_text(r['text'])[:max_text_tokens]
        if not text_ids or not stream:
            skipped += 1
            continue

        input_ids = [lang_id] + text_ids + [TTS] + stream + [EOS]
        labels = [-100] * (1 + len(text_ids) + 1) + stream + [EOS]
        samples.append({
            'input_ids': input_ids, 'labels': labels,
            'lang': r.get('lang', 'en'), 'text': r['text'],
            'mode': 't2s',
        })
    log(f'  t2s samples={len(samples)} skipped={skipped}')
    return samples


def build_roundtrip_samples(
    rows: List[Dict],
    encoder,
    codec: MimiCodec,
    tokenizer: DuplexTokenizer,
    max_audio_frames: int = 200,
    stride: int = 4,
    max_codec_frames: int = 250,
    max_text_tokens: int = 200,
    codec_cache: str = '',
) -> List[Dict]:
    """[audio] <transcribe> text <tts> codec-of-same-audio (ablation only)."""
    from v11.duplex.tokenizer import AUDIO_PAD, ENV_MARK, TRANSCRIBE

    vocab = tokenizer.vocab
    samples: List[Dict] = []
    for r in rows:
        audio = r['audio']
        wf = torch.tensor(audio['array'], dtype=torch.float32)
        sr = int(audio['sampling_rate'])
        frames = encoder.encode_frames(wf, sr, stride=stride,
                                       max_frames=max_audio_frames).detach().cpu().float()
        T = frames.shape[0]
        codes = codec.encode_cached(
            wf, sr, cache_dir=codec_cache or None,
            extra_key=str(r.get('text', ''))[:80],
        )
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
        for t in text_ids:
            input_ids.append(t)
            labels.append(t)
        input_ids.append(TTS)
        labels.append(-100)
        for c in stream:
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


def build_samples(task: str, rows, encoder, codec, tokenizer, **kw) -> List[Dict]:
    task = task.lower()
    if task == 't2s':
        return build_tts_samples(
            rows, codec, tokenizer,
            max_codec_frames=kw.get('max_codec_frames', 250),
            max_text_tokens=kw.get('max_text_tokens', 200),
            codec_cache=kw.get('codec_cache', ''),
        )
    if task == 'both':
        from v11.duplex.train_asr import build_asr_samples
        s2t = build_asr_samples(rows, encoder, tokenizer,
                                max_audio_frames=kw.get('max_audio_frames', 200),
                                stride=kw.get('stride', 4))
        for s in s2t:
            s['mode'] = 's2t'
        t2s = build_tts_samples(
            rows, codec, tokenizer,
            max_codec_frames=kw.get('max_codec_frames', 250),
            codec_cache=kw.get('codec_cache', ''),
        )
        return s2t + t2s
    if task == 'roundtrip':
        return build_roundtrip_samples(
            rows, encoder, codec, tokenizer,
            max_audio_frames=kw.get('max_audio_frames', 200),
            stride=kw.get('stride', 4),
            max_codec_frames=kw.get('max_codec_frames', 250),
            codec_cache=kw.get('codec_cache', ''),
        )
    raise ValueError(f'Unknown task {task!r}. Use t2s | both | roundtrip.')


class TTSDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_tts(batch: List[Dict]) -> Dict:
    """Pad token sequences. t2s samples have no Whisper frames."""
    b = len(batch)
    max_len = max(len(s['input_ids']) for s in batch)
    input_ids = torch.full((b, max_len), 0, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    for i, s in enumerate(batch):
        n = len(s['input_ids'])
        input_ids[i, :n] = torch.tensor(s['input_ids'], dtype=torch.long)
        labels[i, :n] = torch.tensor(s['labels'], dtype=torch.long)
    return {'input_ids': input_ids, 'labels': labels, 'raw_batch': batch}


# ── Codec generation (masked to codec id-space) ──────────────────────────────

@torch.no_grad()
def generate_speech(
    model,
    text: str,
    lang: str,
    tokenizer: DuplexTokenizer,
    device: torch.device,
    max_codec_tokens: int = 400,
    min_codec_tokens: int = 32,
) -> torch.Tensor:
    """Greedy text->codec generation, constrained to codec ids (+eos). Returns codes [K,T].

    EOS is blocked for the first `min_codec_tokens` steps so a weakly trained
    model cannot immediately emit eos (one token vs 8k spread codec ids).
    """
    model.eval()
    vocab = tokenizer.vocab
    lang_id = LANG_TOKEN.get(str(lang).lower(), LANG_TOKEN['en'])
    text_ids = tokenizer.encode_text(text)
    prompt = [lang_id] + text_ids + [TTS]
    ids = torch.tensor([prompt], dtype=torch.long, device=device)

    codec_lo, codec_hi = vocab.codec_offset, vocab.total_size
    allow = torch.full((vocab.total_size,), float('-inf'), device=device)
    allow[codec_lo:codec_hi] = 0.0

    stream: List[int] = []
    recurrent = bool(getattr(model, 'supports_recurrent', False))
    max_seq = int(getattr(model.config, 'max_seq_len', 2048))

    def _pick(logits_last: torch.Tensor, n_so_far: int) -> torch.Tensor:
        masked = logits_last + allow
        if n_so_far < min_codec_tokens:
            masked = masked.clone()
            masked[..., EOS] = float('-inf')
        else:
            masked = masked.clone()
            masked[..., EOS] = 0.0
        return masked.argmax(-1, keepdim=True)

    if recurrent:
        logits, states, _ = model(ids)
        step = ids.shape[1]
        nxt = _pick(logits[:, -1], 0)
        for _ in range(max_codec_tokens):
            tid = int(nxt.item())
            if tid == EOS:
                break
            stream.append(tid)
            logits, states, _ = model(nxt, states=states, step_offset=step)
            step += 1
            nxt = _pick(logits[:, -1], len(stream))
    else:
        for _ in range(max_codec_tokens):
            logits, _, _ = model(ids)
            nxt = _pick(logits[:, -1], len(stream))
            tid = int(nxt.item())
            if tid == EOS:
                break
            stream.append(tid)
            ids = torch.cat([ids, nxt], dim=1)
            if ids.size(1) >= max_seq:
                break
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
        logits, _, _ = model(input_ids)
        total_loss += V11DuplexLM.compute_loss(logits, labels).item()
        nb += 1
        pred = logits[:, :-1].argmax(-1)
        tgt = labels[:, 1:]
        mask = tgt != -100
        correct += (pred[mask] == tgt[mask]).sum().item()
        total += int(mask.sum().item())
    return {'loss': total_loss / max(1, nb), 'token_acc': correct / max(1, total)}


@torch.no_grad()
def save_prompt_wavs(model, tokenizer, codec, device, out_dir: Path,
                     max_codec_tokens: int = 400) -> None:
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang, text in LISTEN_PROMPTS:
        codes = generate_speech(model, text, lang, tokenizer, device,
                                max_codec_tokens=max_codec_tokens)
        path = out_dir / f'{lang}.wav'
        if codes.shape[1] == 0:
            log(f'  wav skip {lang}: empty codec stream')
            continue
        wav = codec.decode(codes)
        sf.write(str(path), wav, 24000)
        log(f'  wrote {path.name} frames={codes.shape[1]} dur={len(wav)/24000:.2f}s')


@torch.no_grad()
def round_trip_wer(model, samples, tokenizer, codec, device, n: int = 8,
                   asr_model_name: str = 'openai/whisper-small') -> Optional[float]:
    """Decode generated speech through Mimi then Whisper ASR; CER vs reference text."""
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
        wav = codec.decode(codes)
        feats = proc(np.asarray(wav), sampling_rate=24000, return_tensors='pt')
        gen = asr.generate(feats.input_features.to(device), max_new_tokens=128)
        hyp = proc.batch_decode(gen, skip_special_tokens=True)[0]
        cers.append(char_error_rate(s['text'], hyp))
    return sum(cers) / max(1, len(cers))


def save_checkpoint(path: Path, model, opt, cfg, epoch, batch_idx, global_step,
                    history, tokenizer_dir, task, metrics=None, extra=None):
    tmp = path.with_name(path.name + '.tmp')
    ckpt = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'config': cfg,
            'epoch': epoch, 'batch_idx': batch_idx, 'global_step': global_step,
            'history': history, 'tokenizer_dir': tokenizer_dir, 'task': task}
    if metrics is not None:
        ckpt['metrics'] = metrics
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def build_backbone(backbone: str, preset: str, vocab_size: int, device):
    if backbone == 'pam':
        cfg = get_duplex_config(preset, vocab_size=vocab_size)
        model = V11DuplexLM(cfg, audio_feat_dim=0).to(device)
        return model, cfg
    if backbone == 'transformer':
        from v11.duplex.transformer_tts import TransformerTTS
        model = TransformerTTS(vocab_size).to(device)
        return model, model.config
    raise ValueError(f'Unknown backbone {backbone!r}. Use pam | transformer.')


def parse_args():
    p = argparse.ArgumentParser(description='V11 duplex Stage B (TTS / T2S)')
    p.add_argument('--preset', default='duplex_100m')
    p.add_argument('--backbone', default='pam', choices=['pam', 'transformer'])
    p.add_argument('--tokenizer_dir', default='checkpoints_v11_duplex_tokenizer')
    p.add_argument('--task', default='t2s', choices=['t2s', 'both', 'roundtrip'])
    p.add_argument('--languages', default='hindi,gujarati')
    p.add_argument('--n_per_lang', type=int, default=2000)
    p.add_argument('--n_english', type=int, default=2000)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--max_audio_frames', type=int, default=200)
    p.add_argument('--max_codec_frames', type=int, default=250)
    p.add_argument('--max_text_tokens', type=int, default=200)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--val_frac', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--whisper', default='openai/whisper-small')
    p.add_argument('--mimi', default='kyutai/mimi')
    p.add_argument('--ckpt_dir', default='')
    p.add_argument('--resume', default='')
    p.add_argument('--init_from', default='', help='optional warm-start (leave empty for TTS-from-scratch)')
    p.add_argument('--codec_cache', default='.cache/mimi_codes')
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--save_every_steps', type=int, default=500)
    p.add_argument('--eval_round_trip', action='store_true')
    p.add_argument('--save_wavs', action='store_true', default=True)
    p.add_argument('--no_save_wavs', action='store_false', dest='save_wavs')
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
    vocab_size = tokenizer.vocab.total_size
    tag = f'{args.backbone}_{args.task}'
    ckpt_dir = Path(args.ckpt_dir or f'checkpoints_v11_{args.preset}_tts_{tag}')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    encoder = None
    if args.task in ('both', 'roundtrip'):
        from v11.duplex.encoder import FrozenWhisperEncoder
        encoder = FrozenWhisperEncoder(args.whisper, device=str(device))
        log('Loaded Whisper encoder (needed for both/roundtrip)')

    codec = MimiCodec(tokenizer.vocab, model_name=args.mimi, device=str(device))
    model, cfg = build_backbone(args.backbone, args.preset, vocab_size, device)
    counts = model.count_parameters()
    log(f'Backbone {args.backbone} preset={args.preset}: {counts} | task={args.task} V={vocab_size}')

    if args.init_from and Path(args.init_from).exists():
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'], strict=False)
        log(f'Warm-started from {args.init_from}')

    langs = [s.strip() for s in args.languages.split(',') if s.strip()]
    langs = [s for s in langs if s.lower() not in ('none', '-', 'off')]
    data_t0 = time.time()
    rows = load_tts_rows(
        languages=langs, n_per_lang=args.n_per_lang,
        include_english=args.n_english > 0, n_english=args.n_english,
        seed=args.seed,
    )
    log(f'Loaded {len(rows)} rows in {elapsed_since(data_t0)}')
    prep_t0 = time.time()
    log(f'Building {args.task} samples (Mimi codes, cache={args.codec_cache})...')
    samples = build_samples(
        args.task, rows, encoder, codec, tokenizer,
        max_audio_frames=args.max_audio_frames, stride=args.stride,
        max_codec_frames=args.max_codec_frames,
        max_text_tokens=args.max_text_tokens,
        codec_cache=args.codec_cache,
    )
    log(f'Built {len(samples)} samples in {elapsed_since(prep_t0)}')
    if len(samples) < 8:
        raise RuntimeError(f'Too few TTS samples ({len(samples)}); check data + Mimi encode.')

    ds = TTSDataset(samples)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_val = min(n_val, len(ds) - 1)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    if args.task == 't2s':
        collate = collate_tts
    else:
        from v11.duplex.train_asr import collate_asr
        feat_dim = encoder.out_dim
        collate = lambda b, fd=feat_dim: collate_asr(b, fd)
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
        log(f'Resumed {resume_path}: ep{start_epoch} step{global_step}')

    stop = {'flag': False}

    def _sig(signum, _f):
        stop['flag'] = True
        log(f'Signal {signum}: saving latest.pt...')
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    extra = {'backbone': args.backbone, 'preset': args.preset}
    best_acc = 0.0
    t0 = time.time()
    log(f'Training start: {len(train_loader)} batches/epoch x {args.epochs} epochs '
        f'train={len(train_ds)} val={len(val_ds)}')
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        epoch_loss, nb = 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            for g in opt.param_groups:
                g['lr'] = args.lr * lr_at(global_step)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            opt.zero_grad(set_to_none=True)
            with _amp_ctx(device, args.amp):
                if args.task != 't2s' and batch.get('frames') is not None:
                    positions = batch['audio_positions'].to(device)
                    frames = batch['frames']
                    embeds = model.project_audio(frames.to(device)) if frames.shape[1] > 0 else None
                    logits, _, _ = model(input_ids, audio_embeds=embeds, audio_positions=positions)
                else:
                    logits, _, _ = model(input_ids)
                loss = V11DuplexLM.compute_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            nb += 1
            global_step += 1
            if batch_idx % args.log_every == 0:
                log(f'ep{epoch} step{global_step} loss={loss.item():.4f} '
                    f'lr={opt.param_groups[0]["lr"]:.2e}')
            if args.save_every_steps and global_step % args.save_every_steps == 0:
                save_checkpoint(ckpt_dir / 'latest.pt', model, opt, cfg, epoch,
                                batch_idx, global_step, history, args.tokenizer_dir, args.task,
                                extra=extra)
                log(f'checkpoint @ step {global_step} (elapsed {elapsed_since(t0)})')
            if stop['flag']:
                save_checkpoint(ckpt_dir / 'latest.pt', model, opt, cfg, epoch,
                                batch_idx, global_step, history, args.tokenizer_dir, args.task,
                                extra=extra)
                log(f'Shutdown: saved latest.pt @ step {global_step}')
                return
        start_batch = 0
        val = codec_token_accuracy(model, val_loader, device)
        row = {'epoch': epoch, 'train_loss': epoch_loss / max(1, nb), **val}
        if args.eval_round_trip:
            wer = round_trip_wer(model, val_ds.dataset.samples, tokenizer, codec, device)
            if wer is not None:
                row['round_trip_cer'] = wer
        history.append(row)
        rt = f' rtCER={row["round_trip_cer"]:.3f}' if 'round_trip_cer' in row else ''
        log(f'=== epoch {epoch} train_loss={row["train_loss"]:.4f} '
            f'val_loss={val["loss"]:.4f} codec_acc={val["token_acc"]:.3f}{rt} '
            f'epoch_time={elapsed_since(epoch_t0)} total={elapsed_since(t0)} ===')
        save_checkpoint(ckpt_dir / 'latest.pt', model, opt, cfg, epoch,
                        len(train_loader) - 1, global_step, history, args.tokenizer_dir,
                        args.task, metrics=row, extra=extra)
        if val['token_acc'] >= best_acc:
            best_acc = val['token_acc']
            torch.save({'model': model.state_dict(), 'config': cfg, 'metrics': row,
                        'tokenizer_dir': args.tokenizer_dir, 'task': args.task,
                        'backbone': args.backbone, 'preset': args.preset},
                       ckpt_dir / 'best_model.pt')
            log(f'New best codec_acc={best_acc:.3f} -> best_model.pt')
        if args.save_wavs:
            wav_dir = ckpt_dir / 'wavs' / f'epoch_{epoch:02d}'
            try:
                save_prompt_wavs(model, tokenizer, codec, device, wav_dir)
            except Exception as e:  # noqa: BLE001
                log(f'  wav dump failed: {type(e).__name__}: {e}')

    elapsed = time.time() - t0
    with open(ckpt_dir / 'metrics.json', 'w') as f:
        json.dump({'stage': 'B_tts', 'task': args.task, 'preset': args.preset,
                   'backbone': args.backbone, 'languages': args.languages,
                   'elapsed_s': elapsed, 'history': history,
                   'best_codec_acc': best_acc, 'n_train': len(train_ds),
                   'n_val': len(val_ds)}, f, indent=2)
    log(f'Done in {elapsed_since(t0)}. best codec_acc={best_acc:.3f}. Ckpts in {ckpt_dir}')


if __name__ == '__main__':
    main()
