"""Stage 0 duplex trainer (self-contained; does not use v7/train.py)."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from v11.duplex.config import get_duplex_config
from v11.duplex.data import SyntheticDuplexDataset, collate_duplex
from v11.duplex.model import V11DuplexLM
from v11.duplex.thinking import VOCAB


def parse_args():
    p = argparse.ArgumentParser(description="V11 duplex Stage 0 (synthetic)")
    p.add_argument('--preset', default='duplex_5m')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n_train', type=int, default=8192)
    p.add_argument('--n_val', type=int, default=1024)
    p.add_argument('--n_blocks', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--ckpt_dir', default='')
    p.add_argument('--log_every', type=int, default=50)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_think_acc = 0.0
    total_think_n = 0
    n_batches = 0
    for input_ids, labels, attn in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits, _, _ = model(input_ids)
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
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    cfg = get_duplex_config(args.preset)
    ckpt_dir = Path(args.ckpt_dir or f'checkpoints_v11_{args.preset}_stage0')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = V11DuplexLM(cfg).to(device)
    params = model.count_parameters()
    print(f"Preset {args.preset}: {params}")

    train_ds = SyntheticDuplexDataset(
        n_samples=args.n_train, n_blocks=args.n_blocks, seed=args.seed,
    )
    val_ds = SyntheticDuplexDataset(
        n_samples=args.n_val, n_blocks=args.n_blocks, seed=args.seed + 1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_duplex,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_duplex,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    think_random = 1.0 / len(VOCAB.thinking_ids)

    history = []
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_think = 0.0
        epoch_think_n = 0
        for batch_idx, (input_ids, labels, _) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _, _ = model(input_ids)
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
                    f"ep{epoch} step{global_step} loss={loss.item():.4f} "
                    f"think_acc={ta:.3f} (random={think_random:.3f})"
                )

        n_train = max(1, len(train_loader))
        train_metrics = {
            'loss': epoch_loss / n_train,
            'think_acc': epoch_think / max(1, epoch_think_n),
        }
        val_metrics = evaluate(model, val_loader, device)
        row = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics}
        history.append(row)
        print(
            f"=== epoch {epoch} train loss={train_metrics['loss']:.4f} "
            f"think={train_metrics['think_acc']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} think={val_metrics['think_acc']:.3f} ==="
        )

    elapsed = time.time() - t0
    summary = {
        'preset': args.preset,
        'params': params,
        'epochs': args.epochs,
        'elapsed_s': elapsed,
        'think_random_baseline': think_random,
        'history': history,
        'final_val_think_acc': history[-1]['val']['think_acc'],
    }
    with open(ckpt_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    torch.save(
        {'model': model.state_dict(), 'config': cfg, 'metrics': summary},
        ckpt_dir / 'best_model.pt',
    )
    print(f"Saved {ckpt_dir / 'best_model.pt'} ({elapsed:.1f}s)")
    print(f"Final val thinking acc: {summary['final_val_think_acc']:.3f} vs random {think_random:.3f}")


if __name__ == '__main__':
    main()
