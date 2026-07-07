"""Stage A: real S2T (ASR) training for the duplex voice model.

Frozen Whisper encoder frames (~12.5 Hz) are injected as complex embeddings at
`<audio_pad>` slots; the PAM backbone autoregressively emits the transcript in
the unified SentencePiece id-space. This is the first stage that makes the model
actually understand speech (vs the turn-taking POC).

Sequence layout per utterance:

    <env> [audio_pad x T] <transcribe> <lang> tok tok ... <eos>
    |------------- masked (-100) -------------|--- CE loss on text + eos ---|

Gate: held-out CER (target < ~15%) before Stage B. Hybrid CTC head on audio
frames forces acoustic grounding alongside the AR head.

Launch (RTX 4090, tmux):
    tmux new-session -d -s asr './scripts/run_v11_duplex_asr.sh'
Resume:
    RESUME=checkpoints_v11_duplex_100m_asr/latest.pt ./scripts/run_v11_duplex_asr.sh
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from v11.duplex.audio_data import load_asr_rows
from v11.duplex.config import get_duplex_config
from v11.duplex.encoder import FrozenWhisperEncoder
from v11.duplex.logutil import elapsed_since, log
from v11.duplex.model import V11DuplexLM
from v11.duplex.tokenizer import (
    AUDIO_PAD, ENV_MARK, EOS, LANG_TOKEN, TRANSCRIBE, DuplexTokenizer,
)


# ── Dataset: precompute frozen encoder frames + token layout ─────────────────

def build_asr_samples(
    rows: List[Dict],
    encoder: FrozenWhisperEncoder,
    tokenizer: DuplexTokenizer,
    max_audio_frames: int = 200,
    stride: int = 4,
    max_text_tokens: int = 200,
) -> List[Dict]:
    """Encode each (audio, text) row into frozen frames + input/label ids.

    Frames are stored on CPU (frozen encoder output); the trainable audio
    projection runs each step, so we never re-run Whisper across epochs.
    """
    samples: List[Dict] = []
    for r in rows:
        audio = r['audio']
        wf = torch.tensor(audio['array'], dtype=torch.float32)
        sr = int(audio['sampling_rate'])
        frames = encoder.encode_frames(wf, sr, stride=stride, max_frames=max_audio_frames)
        frames = frames.detach().cpu().float()
        T = frames.shape[0]
        lang_id = LANG_TOKEN.get(str(r.get('lang', 'en')).lower(), LANG_TOKEN['en'])
        text_ids = tokenizer.encode_text(r['text'])[:max_text_tokens]
        if not text_ids:
            continue
        text_local = tokenizer.sp.encode(r['text'], out_type=int)
        if len(text_local) > max_text_tokens:
            text_local = text_local[:max_text_tokens]
        if not text_local:
            continue
        # CTC needs T >= len(target); truncate target if clip is very short.
        if T < len(text_local):
            text_local = text_local[:max(1, T)]

        input_ids: List[int] = [ENV_MARK] + [AUDIO_PAD] * T + [TRANSCRIBE, lang_id]
        labels: List[int] = [-100] * len(input_ids)
        audio_positions = list(range(1, 1 + T))  # AUDIO_PAD slots (after ENV_MARK)
        for t in text_ids:
            input_ids.append(t)
            labels.append(t)
        input_ids.append(EOS)
        labels.append(EOS)

        samples.append({
            'input_ids': input_ids,
            'labels': labels,
            'audio_positions': audio_positions,
            'frames': frames,
            'lang': r.get('lang', 'en'),
            'text': r['text'],
            'text_local': text_local,
            'n_frames': T,
        })
    return samples


class ASRDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_asr(batch: List[Dict], feat_dim: int) -> Dict:
    b = len(batch)
    max_len = max(len(s['input_ids']) for s in batch)
    max_T = max(s['frames'].shape[0] for s in batch)
    input_ids = torch.full((b, max_len), 0, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    frames = torch.zeros(b, max_T, feat_dim, dtype=torch.float32)
    audio_positions = torch.full((b, max_T), -1, dtype=torch.long)
    ctc_target_lengths = torch.zeros(b, dtype=torch.long)
    ctc_chunks: List[torch.Tensor] = []
    for i, s in enumerate(batch):
        n = len(s['input_ids'])
        input_ids[i, :n] = torch.tensor(s['input_ids'], dtype=torch.long)
        labels[i, :n] = torch.tensor(s['labels'], dtype=torch.long)
        T = s['frames'].shape[0]
        frames[i, :T] = s['frames']
        for j, pos in enumerate(s['audio_positions']):
            audio_positions[i, j] = pos
        local = s['text_local']
        ctc_target_lengths[i] = len(local)
        ctc_chunks.append(torch.tensor(local, dtype=torch.long))
    ctc_targets = torch.cat(ctc_chunks) if ctc_chunks else torch.zeros(0, dtype=torch.long)
    return {
        'input_ids': input_ids, 'labels': labels,
        'frames': frames, 'audio_positions': audio_positions,
        'ctc_targets': ctc_targets,
        'ctc_target_lengths': ctc_target_lengths,
        'raw_batch': batch,
    }


# ── Greedy transcription (audio prefill + state-carry decode) ────────────────

@torch.no_grad()
def greedy_transcribe(
    model: V11DuplexLM,
    frames: torch.Tensor,
    lang: str,
    tokenizer: DuplexTokenizer,
    device: torch.device,
    max_new: int = 160,
) -> str:
    model.eval()
    T = frames.shape[0]
    lang_id = LANG_TOKEN.get(str(lang).lower(), LANG_TOKEN['en'])
    prompt = [ENV_MARK] + [AUDIO_PAD] * T + [TRANSCRIBE, lang_id]
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    audio_embeds = model.project_audio(frames.unsqueeze(0).to(device))
    positions = torch.tensor([list(range(1, 1 + T))], dtype=torch.long, device=device)
    logits, states, _ = model(ids, audio_embeds=audio_embeds, audio_positions=positions)
    step = ids.shape[1]
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out: List[int] = []
    for _ in range(max_new):
        tid = int(nxt)
        if tid == EOS:
            break
        out.append(tid)
        logits, states, _ = model(nxt, states=states, step_offset=step)
        step += 1
        nxt = logits[:, -1].argmax(-1, keepdim=True)
    return tokenizer.decode_text(out)


@torch.no_grad()
def ctc_greedy_decode(
    model: V11DuplexLM,
    frames: torch.Tensor,
    lang: str,
    tokenizer: DuplexTokenizer,
    device: torch.device,
) -> str:
    """Frame-wise CTC greedy decode: reads directly from audio-frame hiddens."""
    model.eval()
    if model.frame_heads is None:
        return ''
    T = frames.shape[0]
    lang_id = LANG_TOKEN.get(str(lang).lower(), LANG_TOKEN['en'])
    prompt = [ENV_MARK] + [AUDIO_PAD] * T + [TRANSCRIBE, lang_id]
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    audio_embeds = model.project_audio(frames.unsqueeze(0).to(device))
    positions = torch.tensor([list(range(1, 1 + T))], dtype=torch.long, device=device)
    _, _, _, hidden = model(
        ids, audio_embeds=audio_embeds, audio_positions=positions, return_hidden=True,
    )
    log_probs, lengths = model.ctc_log_probs(hidden, positions)
    blank = model.frame_heads.blank_id
    preds = log_probs[0, : int(lengths[0])].argmax(-1).tolist()
    collapsed: List[int] = []
    prev = blank
    for p in preds:
        if p != blank and p != prev:
            collapsed.append(p)
        prev = p
    gids = [tokenizer.vocab.text_to_global(p) for p in collapsed]
    return tokenizer.decode_text(gids)


def _collapse_ctc_preds(preds: List[int], blank: int) -> List[int]:
    out: List[int] = []
    prev = blank
    for p in preds:
        if p != blank and p != prev:
            out.append(p)
        prev = p
    return out


def _edit_distance(a: List, b: List) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def char_error_rate(ref: str, hyp: str) -> float:
    ref_c, hyp_c = list(ref.strip()), list(hyp.strip())
    if not ref_c:
        return 0.0 if not hyp_c else 1.0
    return _edit_distance(ref_c, hyp_c) / len(ref_c)


# ── Train / eval ─────────────────────────────────────────────────────────────

def _audio_embeds(model, frames, device):
    return model.project_audio(frames.to(device))


def _ctc_loss(
    model: V11DuplexLM,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    ctc_targets: torch.Tensor,
    ctc_target_lengths: torch.Tensor,
) -> torch.Tensor:
    log_probs, input_lengths = model.ctc_log_probs(hidden, positions)
    if int(input_lengths.max().item()) == 0:
        return log_probs.sum() * 0.0
    return F.ctc_loss(
        log_probs.transpose(0, 1),
        ctc_targets,
        input_lengths,
        ctc_target_lengths,
        blank=model.frame_heads.blank_id,
        zero_infinity=True,
    )
    return model.project_audio(frames.to(device))


def _audio_proj_grad_norm(model) -> float:
    """L2 norm of gradients on the trainable audio projection (0 signals the
    projection is detached from the loss -> the audio path is not learning)."""
    if model.audio_proj is None:
        return 0.0
    sq = sum(
        float(p.grad.detach().float().norm().item()) ** 2
        for p in model.audio_proj.parameters() if p.grad is not None
    )
    return sq ** 0.5


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, cer_samples: int = 32,
             n_show: int = 3) -> Dict:
    model.eval()
    total_loss, total_ctc, n_batches = 0.0, 0.0, 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        positions = batch['audio_positions'].to(device)
        ctc_targets = batch['ctc_targets'].to(device)
        ctc_target_lengths = batch['ctc_target_lengths'].to(device)
        embeds = _audio_embeds(model, batch['frames'], device)
        logits, _, _, hidden = model(
            input_ids, audio_embeds=embeds, audio_positions=positions, return_hidden=True,
        )
        total_loss += V11DuplexLM.compute_loss(logits, labels).item()
        if model.frame_heads is not None:
            total_ctc += _ctc_loss(
                model, hidden, positions, ctc_targets, ctc_target_lengths,
            ).item()
        n_batches += 1
    # CER on a subset (greedy decode is sequential -> keep it small)
    cers, ctc_cers, seen = [], [], 0
    per_lang: Dict[str, List[float]] = {}
    per_lang_ctc: Dict[str, List[float]] = {}
    samples: List[Dict] = []
    for batch in loader:
        for s in batch['raw_batch']:
            if seen >= cer_samples:
                break
            hyp = greedy_transcribe(model, s['frames'], s['lang'], tokenizer, device)
            hyp_ctc = ctc_greedy_decode(model, s['frames'], s['lang'], tokenizer, device)
            cer = char_error_rate(s['text'], hyp)
            ctc_cer = char_error_rate(s['text'], hyp_ctc)
            cers.append(cer)
            ctc_cers.append(ctc_cer)
            per_lang.setdefault(str(s['lang']), []).append(cer)
            per_lang_ctc.setdefault(str(s['lang']), []).append(ctc_cer)
            if len(samples) < n_show:
                samples.append({
                    'lang': s['lang'], 'ref': s['text'],
                    'hyp': hyp, 'hyp_ctc': hyp_ctc,
                })
            seen += 1
        if seen >= cer_samples:
            break
    per_lang_cer = {k: sum(v) / len(v) for k, v in per_lang.items()}
    per_lang_ctc_cer = {k: sum(v) / len(v) for k, v in per_lang_ctc.items()}
    return {
        'loss': total_loss / max(1, n_batches),
        'ctc_loss': total_ctc / max(1, n_batches),
        'cer': sum(cers) / max(1, len(cers)),
        'ctc_cer': sum(ctc_cers) / max(1, len(ctc_cers)),
        'cer_n': len(cers),
        'per_lang_cer': per_lang_cer,
        'per_lang_ctc_cer': per_lang_ctc_cer,
        'samples': samples,
    }


def save_checkpoint(path: Path, model, opt, cfg, epoch, batch_idx, global_step,
                    history, tokenizer_dir, metrics=None):
    tmp = path.with_name(path.name + '.tmp')
    ckpt = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
            'config': cfg, 'epoch': epoch, 'batch_idx': batch_idx,
            'global_step': global_step, 'history': history,
            'tokenizer_dir': tokenizer_dir}
    if metrics is not None:
        ckpt['metrics'] = metrics
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description='V11 duplex Stage A (ASR / S2T)')
    p.add_argument('--preset', default='duplex_100m')
    p.add_argument('--tokenizer_dir', default='checkpoints_v11_duplex_tokenizer')
    p.add_argument('--languages', default='hindi,gujarati')
    p.add_argument('--n_per_lang', type=int, default=2000)
    p.add_argument('--n_english', type=int, default=2000)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--max_audio_frames', type=int, default=200)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--val_frac', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--whisper', default='openai/whisper-small')
    p.add_argument('--ckpt_dir', default='')
    p.add_argument('--resume', default='')
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--save_every_steps', type=int, default=500)
    p.add_argument('--cer_samples', type=int, default=32)
    p.add_argument('--ctc_weight', type=float, default=0.5,
                   help='Weight on CTC alignment loss (0 disables CTC head training)')
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
    ckpt_dir = Path(args.ckpt_dir or f'checkpoints_v11_{args.preset}_asr')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log(f'Tokenizer: {args.tokenizer_dir} (vocab {tokenizer.vocab.total_size})')
    encoder = FrozenWhisperEncoder(args.whisper, device=str(device))
    model = V11DuplexLM(
        cfg, audio_feat_dim=encoder.out_dim, n_text=tokenizer.vocab.n_text,
    ).to(device)
    log(f'Preset {args.preset}: {model.count_parameters()}')

    langs = [s.strip() for s in args.languages.split(',') if s.strip()]
    log(
        f'Data: Kathbath {langs} ({args.n_per_lang}/lang)'
        + (f' + LibriSpeech english ({args.n_english})' if args.n_english > 0 else ' (no english)')
    )
    data_t0 = time.time()
    rows = load_asr_rows(
        languages=langs, n_per_lang=args.n_per_lang,
        include_english=args.n_english > 0, n_english=args.n_english, seed=args.seed,
    )
    log(f'Loaded {len(rows)} rows in {elapsed_since(data_t0)}')
    prep_t0 = time.time()
    log('Encoding frozen Whisper frames (one-time)...')
    samples = build_asr_samples(rows, encoder, tokenizer,
                                max_audio_frames=args.max_audio_frames, stride=args.stride)
    log(f'Built {len(samples)} ASR samples in {elapsed_since(prep_t0)}')
    ds = ASRDataset(samples)
    n_val = max(1, int(len(ds) * args.val_frac))
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
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
        log(f'Resumed {resume_path}: ep{start_epoch} batch{start_batch} step{global_step}')

    stop = {'flag': False}

    def _sig(signum, _f):
        stop['flag'] = True
        log(f'Signal {signum}: saving latest.pt at next boundary...')
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    best_cer = float('inf')
    best_gate_cer = float('inf')
    t0 = time.time()
    log(f'Training start: {len(train_loader)} batches/epoch x {args.epochs} epochs '
        f'(ctc_weight={args.ctc_weight})')
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        epoch_loss, nb = 0.0, 0
        audio_grad_norm = 0.0
        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            for g in opt.param_groups:
                g['lr'] = args.lr * lr_at(global_step)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            positions = batch['audio_positions'].to(device)
            ctc_targets = batch['ctc_targets'].to(device)
            ctc_target_lengths = batch['ctc_target_lengths'].to(device)
            opt.zero_grad(set_to_none=True)
            with _amp_ctx(device, args.amp):
                # project_audio is trainable and must receive gradients (the frozen
                # Whisper frames were already precomputed under no_grad upstream).
                embeds = _audio_embeds(model, batch['frames'], device)
                logits, _, _, hidden = model(
                    input_ids, audio_embeds=embeds, audio_positions=positions,
                    return_hidden=True,
                )
                ar_loss = V11DuplexLM.compute_loss(logits, labels)
                if args.ctc_weight > 0 and model.frame_heads is not None:
                    ctc_loss = _ctc_loss(
                        model, hidden, positions, ctc_targets, ctc_target_lengths,
                    )
                    loss = ar_loss + args.ctc_weight * ctc_loss
                else:
                    ctc_loss = torch.tensor(0.0, device=device)
                    loss = ar_loss
            loss.backward()
            audio_grad_norm = _audio_proj_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            nb += 1
            global_step += 1
            if batch_idx % args.log_every == 0:
                log(f'ep{epoch} step{global_step} loss={loss.item():.4f} '
                    f'ar={ar_loss.item():.4f} ctc={ctc_loss.item():.4f} '
                    f'ppl={math.exp(min(20, loss.item())):.2f} lr={opt.param_groups[0]["lr"]:.2e}')
            if args.save_every_steps and global_step % args.save_every_steps == 0:
                save_checkpoint(ckpt_dir / 'latest.pt', model, opt, cfg, epoch,
                                batch_idx, global_step, history, args.tokenizer_dir)
                log(f'checkpoint @ step {global_step} (elapsed {elapsed_since(t0)})')
            if stop['flag']:
                save_checkpoint(ckpt_dir / 'latest.pt', model, opt, cfg, epoch,
                                batch_idx, global_step, history, args.tokenizer_dir)
                log(f'Shutdown: saved latest.pt @ step {global_step}')
                return
        start_batch = 0
        val = evaluate(model, val_loader, tokenizer, device, cer_samples=args.cer_samples)
        samples = val.pop('samples', [])
        row = {'epoch': epoch, 'train_loss': epoch_loss / max(1, nb),
               'audio_grad_norm': audio_grad_norm, **val}
        history.append(row)
        per_lang_str = ' '.join(
            f'{k[:2]}={v:.3f}' for k, v in sorted(val.get('per_lang_cer', {}).items())
        )
        per_lang_ctc_str = ' '.join(
            f'{k[:2]}={v:.3f}' for k, v in sorted(val.get('per_lang_ctc_cer', {}).items())
        )
        gate_cer = min(val['cer'], val.get('ctc_cer', val['cer']))
        log(f'=== epoch {epoch} train_loss={row["train_loss"]:.4f} '
            f'val_loss={val["loss"]:.4f} ctc_loss={val.get("ctc_loss", 0):.4f} '
            f'CER={val["cer"]:.3f} ctc_CER={val.get("ctc_cer", val["cer"]):.3f} '
            f'(n={val["cer_n"]}) AR[{per_lang_str}] CTC[{per_lang_ctc_str}] '
            f'audio_grad={audio_grad_norm:.3e} '
            f'epoch_time={elapsed_since(epoch_t0)} total={elapsed_since(t0)} ===')
        for s in samples:
            log(f'  [{s["lang"]}] ref: {s["ref"]}')
            log(f'  [{s["lang"]}] hyp_ar: {s["hyp"]}')
            if 'hyp_ctc' in s:
                log(f'  [{s["lang"]}] hyp_ctc: {s["hyp_ctc"]}')
        save_checkpoint(ckpt_dir / 'latest.pt', model, opt, cfg, epoch,
                        len(train_loader) - 1, global_step, history, args.tokenizer_dir,
                        metrics=row)
        if gate_cer <= best_gate_cer:
            best_gate_cer = gate_cer
            torch.save({'model': model.state_dict(), 'config': cfg, 'metrics': row,
                        'tokenizer_dir': args.tokenizer_dir}, ckpt_dir / 'best_model.pt')
            log(f'New best gate_CER={best_gate_cer:.3f} (AR={val["cer"]:.3f} '
                f'CTC={val.get("ctc_cer", val["cer"]):.3f}) -> best_model.pt')
        if val['cer'] <= best_cer:
            best_cer = val['cer']

    elapsed = time.time() - t0
    with open(ckpt_dir / 'metrics.json', 'w') as f:
        json.dump({'stage': 'A_asr', 'preset': args.preset, 'languages': args.languages,
                   'n_per_lang': args.n_per_lang, 'n_english': args.n_english,
                   'ctc_weight': args.ctc_weight,
                   'elapsed_s': elapsed, 'history': history,
                   'best_cer': best_cer, 'best_gate_cer': best_gate_cer}, f, indent=2)
    log(f'Done in {elapsed_since(t0)}. best gate_CER={best_gate_cer:.3f} '
        f'(AR={best_cer:.3f}). Ckpts in {ckpt_dir}')


if __name__ == '__main__':
    main()
