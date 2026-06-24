"""Stage 1 trainer: frozen Whisper + LibriSpeech (Fisher when LDC available)."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from v11.duplex.audio_data import (
    DEFAULT_STAGE1_DATASET,
    LIBRISPEECH_HF,
    LibriSpeechDuplexDataset,
    collate_stage1,
)
from v11.duplex.config import get_duplex_config
from v11.duplex.encoder import FrozenWhisperEncoder
from v11.duplex.model import V11DuplexLM
from v11.duplex.thinking import VOCAB


def parse_args():
    p = argparse.ArgumentParser(description='V11 duplex Stage 1 (audio + LibriSpeech fallback)')
    p.add_argument('--preset', default='duplex_5m')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--n_pairs', type=int, default=800)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--ckpt_dir', default='')
    p.add_argument('--resume', default='')
    p.add_argument('--whisper', default='openai/whisper-small')
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--dataset', default=DEFAULT_STAGE1_DATASET,
                   help='librispeech (open) | fisher (LDC — not implemented)')
    return p.parse_args()


def batch_audio_embeds(
    model: V11DuplexLM,
    encoder: FrozenWhisperEncoder,
    raw_batch,
    audio_positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build [B, max_slots, dim, 2] audio embeddings aligned to slot count."""
    b, max_slots = audio_positions.shape
    dim = model.config.dim
    out = torch.zeros(b, max_slots, dim, 2, device=device)
    for i, sample in enumerate(raw_batch):
        n_slots = len(sample['audio_slot_indices'])
        if n_slots == 0:
            continue
        audio = sample['user_audio']
        wf = torch.tensor(audio['array'], dtype=torch.float32)
        sr = int(audio['sampling_rate'])
        chunks = encoder.encode_waveform(wf, sr, max_chunks=n_slots)
        chunks = chunks.to(device)
        for j in range(min(n_slots, chunks.shape[0])):
            feat = chunks[j:j + 1]
            if model.audio_proj is not None:
                z = model.project_audio(feat.unsqueeze(0)).squeeze(0)
            else:
                raise RuntimeError('model needs audio_feat_dim set for stage 1')
            out[i, j] = z[0]
    return out


@torch.no_grad()
def evaluate(model, encoder, loader, device):
    model.eval()
    total_loss = 0.0
    total_think_acc = 0.0
    total_think_n = 0
    n_batches = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        audio_positions = batch['audio_positions'].to(device)
        audio_embeds = batch_audio_embeds(
            model, encoder, batch['raw_batch'], audio_positions, device,
        )
        logits, _, _ = model(
            input_ids, labels=labels,
            audio_embeds=audio_embeds, audio_positions=audio_positions,
        )
        loss = V11DuplexLM.compute_loss(logits, labels)
        acc, n = model.thinking_accuracy(logits, labels, VOCAB.thinking_ids)
        total_loss += loss.item()
        if n > 0:
            total_think_acc += acc * n
            total_think_n += n
        n_batches += 1
    if n_batches == 0:
        return {'loss': 0.0, 'think_acc': 0.0}
    return {
        'loss': total_loss / n_batches,
        'think_acc': total_think_acc / max(1, total_think_n),
    }


def main():
    args = parse_args()
    if args.dataset == 'fisher':
        raise NotImplementedError(
            'Fisher LDC97S62 requires LDC license. Use --dataset librispeech (default) '
            'or place Fisher data locally and add a loader in v11/duplex/audio_data.py.'
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = get_duplex_config(args.preset)
    ckpt_dir = Path(args.ckpt_dir or f'checkpoints_v11_{args.preset}_stage1')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f'Stage 1 dataset: {args.dataset} ({LIBRISPEECH_HF} pairs, Fisher=LDC fallback documented)')
    encoder = FrozenWhisperEncoder(args.whisper, device=str(device))
    model = V11DuplexLM(cfg, audio_feat_dim=encoder.out_dim).to(device)
    params = model.count_parameters()
    print(f'Preset {args.preset}: {params}')

    full_ds = LibriSpeechDuplexDataset(n_pairs=args.n_pairs, seed=args.seed)
    n_val = max(1, int(len(full_ds) * args.val_frac))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_stage1, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_stage1, num_workers=0,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        if 'optimizer' in ckpt:
            opt.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f'Resumed from {args.resume} at epoch {start_epoch}')

    think_random = 1.0 / len(VOCAB.thinking_ids)
    history = []
    global_step = 0
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        encoder.encoder.eval()
        epoch_loss = 0.0
        epoch_think = 0.0
        epoch_think_n = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            audio_positions = batch['audio_positions'].to(device)
            with torch.no_grad():
                audio_embeds = batch_audio_embeds(
                    model, encoder, batch['raw_batch'], audio_positions, device,
                )
            opt.zero_grad(set_to_none=True)
            logits, _, _ = model(
                input_ids, audio_embeds=audio_embeds, audio_positions=audio_positions,
            )
            loss = V11DuplexLM.compute_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            acc, n = model.thinking_accuracy(logits, labels, VOCAB.thinking_ids)
            epoch_loss += loss.item()
            if n > 0:
                epoch_think += acc * n
                epoch_think_n += n
            global_step += 1

            if batch_idx % args.log_every == 0:
                ta = acc if n > 0 else 0.0
                print(
                    f'ep{epoch} step{global_step} loss={loss.item():.4f} '
                    f'think_acc={ta:.3f} (random={think_random:.3f})'
                )

        n_train_batches = max(1, len(train_loader))
        train_metrics = {
            'loss': epoch_loss / n_train_batches,
            'think_acc': epoch_think / max(1, epoch_think_n),
        }
        val_metrics = evaluate(model, encoder, val_loader, device)
        row = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics}
        history.append(row)
        print(
            f'=== epoch {epoch} train loss={train_metrics["loss"]:.4f} '
            f'think={train_metrics["think_acc"]:.3f} | '
            f'val loss={val_metrics["loss"]:.4f} think={val_metrics["think_acc"]:.3f} ==='
        )

        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'config': cfg,
                'epoch': epoch,
                'metrics': row,
            },
            ckpt_dir / 'latest.pt',
        )

    elapsed = time.time() - t0
    summary = {
        'stage': 1,
        'dataset': args.dataset,
        'hf_source': LIBRISPEECH_HF,
        'fisher_primary': 'LDC97S62 (requires LDC license — not used unless you add it)',
        'preset': args.preset,
        'params': params,
        'epochs': args.epochs,
        'elapsed_s': elapsed,
        'history': history,
        'final_val_think_acc': history[-1]['val']['think_acc'] if history else 0.0,
    }
    with open(ckpt_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    torch.save(
        {'model': model.state_dict(), 'config': cfg, 'metrics': summary},
        ckpt_dir / 'best_model.pt',
    )
    print(f'Saved {ckpt_dir / "best_model.pt"} ({elapsed:.1f}s)')


if __name__ == '__main__':
    main()
