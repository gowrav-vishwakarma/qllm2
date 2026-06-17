"""
V11 training entrypoint.

Reuses the V7 trainer (`v7.train.V7Trainer`) and the V7 data pipeline
(`v7.data`) unchanged -- only the model and the new memory-dynamics flags are
V11-specific. Every flag defaults to the V7 7d baseline so each run is a clean
single-variable ablation.

Usage:
    .venv/bin/python -m v11.train --preset tiny --epochs 1 --max_samples 200 \
        --dataset tinystories --num_workers 2 --gen_every 0           # smoke
    .venv/bin/python -m v11.train --preset v11_e1_perchannel --batch_size 18 \
        --chunk_size 256 --compile --no_grad_ckpt                     # E1 full
"""

import argparse
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from v11.model import V11LM, get_config, PRESETS
from v7.train import (
    V7Trainer, seed_everything, _notify_training_failure, _is_oom_error,
    _notify_discord, _notify_discord_long,
)
from v7.data import load_wikitext103, load_tinystories, TeeLogger


def build_argparser():
    p = argparse.ArgumentParser(description='V11 PAM (new memory dynamics) training')
    p.add_argument('--preset', type=str, default='v11_baseline', choices=list(PRESETS.keys()))
    p.add_argument('--dataset', type=str, default='wikitext103',
                   choices=['wikitext103', 'tinystories'])
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=18)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--seq_len', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--gradient_clip', type=float, default=1.0)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--compile_mode', type=str, default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--amp_dtype', type=str, default='auto', choices=['auto', 'bf16', 'fp16'])
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--max_samples', type=int, default=9999999)
    p.add_argument('--gen_every', type=int, default=5000)
    p.add_argument('--gen_prompt', type=str, default='In 1923 , the University of')
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_v11')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_grad_ckpt', action='store_true')
    p.add_argument('--chunk_size', type=int, default=None)
    p.add_argument('--activation', type=str, default=None,
                   choices=['modrelu', 'swish', 'phase_mod'])
    # New memory-dynamics overrides (else taken from preset).
    p.add_argument('--decay_mode', type=str, default=None, choices=['head', 'per_channel'])
    p.add_argument('--write_mode', type=str, default=None, choices=['additive', 'delta'])
    p.add_argument('--n_states', type=int, default=None)
    p.add_argument('--delta_chunk', type=int, default=None)
    return p


def main():
    args = build_argparser().parse_args()

    # Discord webhook (optional, same convention as v7).
    env_path = Path(__file__).resolve().parent.parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and line.startswith('DISCORD_HOOK='):
                os.environ['DISCORD_HOOK'] = line.split('=', 1)[1].strip().strip('\'"')
    if os.environ.get('DISCORD_HOOK'):
        print('[Discord] Webhook configured -- notifications enabled', file=sys.stderr)
    else:
        print('[Discord] No webhook (set DISCORD_HOOK in .env to enable)', file=sys.stderr)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'v11_{args.preset}_{args.dataset}.log'
    tee = TeeLogger(log_path, mode='a' if args.resume else 'w')
    sys.stdout = tee
    print(f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('=' * 60)
    print('  V11: PAM with new memory dynamics')
    print(f'  Preset: {args.preset} | Dataset: {args.dataset}')
    print('=' * 60)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(args.seed)

    cfg = get_config(args.preset)
    if args.seq_len is not None:
        cfg.max_seq_len = args.seq_len
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.no_grad_ckpt:
        cfg.gradient_checkpointing = False
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    if args.activation is not None:
        cfg.activation = args.activation
    if args.decay_mode is not None:
        cfg.decay_mode = args.decay_mode
    if args.write_mode is not None:
        cfg.write_mode = args.write_mode
    if args.n_states is not None:
        cfg.n_states = args.n_states
    if args.delta_chunk is not None:
        cfg.delta_chunk = args.delta_chunk

    print(f"\nConfig: {asdict(cfg)}")
    print(f"Memory dynamics: decay_mode={cfg.decay_mode}, write_mode={cfg.write_mode}, "
          f"n_states={cfg.n_states}, chunk_size={cfg.chunk_size}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")

    seq_len = cfg.max_seq_len
    max_samples = args.max_samples if args.max_samples < 9999999 else None
    print(f"\nLoading {args.dataset} (seq_len={seq_len})...")
    if args.dataset == 'wikitext103':
        train_ds, val_ds, tokenizer = load_wikitext103(max_samples=max_samples, seq_len=seq_len)
    else:
        train_ds, val_ds, tokenizer = load_tinystories(
            max_samples=max_samples or 20000, seq_len=seq_len)

    from torch.utils.data import DataLoader
    use_cuda = torch.cuda.is_available()
    nw = args.num_workers if use_cuda else 0
    dl_kwargs = {}
    if nw > 0:
        dl_kwargs['persistent_workers'] = True
        dl_kwargs['prefetch_factor'] = 4
    gen = torch.Generator(); gen.manual_seed(args.seed)

    def _wi(wid):
        np.random.seed(args.seed + wid); random.seed(args.seed + wid)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=use_cuda, generator=gen,
                              worker_init_fn=_wi, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=use_cuda, worker_init_fn=_wi, **dl_kwargs)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = V11LM(cfg)
    params = model.count_parameters()
    print(f"\nModel parameters: {params}")
    print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")

    start_epoch = 0
    checkpoint = None
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1

    trainer = V7Trainer(
        model, train_loader, val_loader, tokenizer,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, gradient_clip=args.gradient_clip,
        max_epochs=args.epochs, checkpoint_dir=args.checkpoint_dir,
        amp_dtype_str=args.amp_dtype, compile_model=args.compile,
        compile_mode=args.compile_mode, gen_every=args.gen_every,
        gen_prompt=args.gen_prompt, log_interval=args.log_interval,
        start_epoch=start_epoch, run_label=f'V11/{args.preset}', log_path=str(log_path),
    )
    if checkpoint and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.global_tokens = checkpoint.get('global_tokens', 0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))

    # Discord "Training started" banner with full config summary (parity with V7).
    _summary = [
        f"Host: {os.uname().nodename}",
        f"Preset: {args.preset} | Dataset: {args.dataset}",
        f"Memory dynamics: decay_mode={cfg.decay_mode} | write_mode={cfg.write_mode} "
        f"| n_states={cfg.n_states} | chunk_size={cfg.chunk_size}",
        f"Complex dim: {cfg.dim} | Layers: {cfg.n_layers} | "
        f"PAM heads={cfg.n_heads} d={cfg.head_dim} | CGU expand={cfg.expand}",
        f"Activation: {cfg.activation} | RoPE: {cfg.use_rope} | GSP: {cfg.use_gsp} | "
        f"grad_ckpt: {cfg.gradient_checkpointing}",
        f"Params: {params['total']:,} ({params['total']/1e6:.1f}M)",
        f"Epochs: {args.epochs} | Batch: {args.batch_size} | "
        f"Batches/epoch: {len(train_loader)} | seq_len={cfg.max_seq_len}",
        f"LR: {args.lr} | warmup={args.warmup_steps} | wd={args.weight_decay} | "
        f"grad_clip={args.gradient_clip} | dropout={cfg.dropout}",
        f"AMP: {args.amp_dtype} | Compile: {args.compile}",
        f"Log: {log_path.resolve()}",
        f"Checkpoint dir: {Path(args.checkpoint_dir).resolve()}",
    ]
    _hdr = '**V11 Training started**' if not args.resume else '**V11 Training resumed**'
    _notify_discord_long(_hdr + '\n```\n' + '\n'.join(_summary) + '\n```')

    trainer.train()
    print(f"\nWall clock end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee._stdout
    tee.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        _notify_training_failure('stopped by user'); raise
    except Exception as e:
        _notify_training_failure('OOM' if _is_oom_error(e) else 'failed', e); raise
