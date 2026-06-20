"""
V11 training entrypoint.

Reuses the V7 trainer (`v7.train.V7Trainer`) with extended data pipeline for
Phase C (DCLM-Edu pretrain + SmolTalk2 SFT).

Usage:
    .venv/bin/python -m v11.train --preset tiny --epochs 1 --max_samples 200 \
        --dataset tinystories --num_workers 2 --gen_every 0           # smoke
    .venv/bin/python -m v11.train --preset v11_e3_k3 --stage pretrain \
        --dataset dclm_edu --token_budget 2000000000 --resume_from ckpt.pt
    .venv/bin/python -m v11.train --preset v11_e3_k3 --stage sft \
        --dataset smoltalk2 --resume_from ckpt.pt --lr 5e-5 --epochs 1
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
from v7.data import (
    load_wikitext103,
    load_tinystories,
    load_dclm_edu,
    load_smoltalk2,
    TeeLogger,
)


def build_argparser():
    p = argparse.ArgumentParser(description='V11 PAM (new memory dynamics) training')
    p.add_argument('--preset', type=str, default='v11_baseline', choices=list(PRESETS.keys()))
    p.add_argument('--dataset', type=str, default='wikitext103',
                   choices=['wikitext103', 'tinystories', 'dclm_edu', 'smoltalk2'])
    p.add_argument('--stage', type=str, default='lm',
                   choices=['lm', 'pretrain', 'sft'],
                   help='lm=legacy WikiText path; pretrain=web stream; sft=chat masked CE')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=18)
    p.add_argument('--lr', type=float, default=None,
                   help='Default: 1e-4 pretrain/lm, 5e-5 sft')
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
    p.add_argument('--token_budget', type=int, default=0,
                   help='Stop after this many training tokens (pretrain pilot)')
    p.add_argument('--edu_score_min', type=int, default=3,
                   help='Minimum edu_int_score for DCLM-Edu rows')
    p.add_argument('--sft_filter', type=str, default='hard', choices=['none', 'hard'])
    p.add_argument('--gen_every', type=int, default=5000)
    p.add_argument('--gen_prompt', type=str, default='In 1923 , the University of')
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints_v11')
    p.add_argument('--resume', type=str, default=None,
                   help='Full resume (model + optimizer + scheduler)')
    p.add_argument('--resume_from', type=str, default=None,
                   help='Load model weights only (fresh optimizer; for SFT stage)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_grad_ckpt', action='store_true')
    p.add_argument('--chunk_size', type=int, default=None)
    p.add_argument('--activation', type=str, default=None,
                   choices=['modrelu', 'swish', 'phase_mod'])
    p.add_argument('--decay_mode', type=str, default=None, choices=['head', 'per_channel'])
    p.add_argument('--write_mode', type=str, default=None, choices=['additive', 'delta'])
    p.add_argument('--n_states', type=int, default=None)
    p.add_argument('--delta_chunk', type=int, default=None)
    return p


def _resolve_stage_dataset(stage: str, dataset: str) -> str:
    if stage == 'pretrain' and dataset == 'wikitext103':
        return 'dclm_edu'
    if stage == 'sft' and dataset == 'wikitext103':
        return 'smoltalk2'
    return dataset


def _resize_embeddings_for_vocab(model, state):
    """Grow loaded token embeddings to the model's vocab (e.g. +ChatML tokens).

    Weights are tied (lm_head reuses the embedding), so only the two complex
    embedding matrices need resizing. New rows are initialized to match
    ``ComplexEmbed`` (normal, std=0.02); existing rows are copied verbatim.
    """
    target_state = model.state_dict()
    for key in ('embed.embed_real.weight', 'embed.embed_imag.weight'):
        if key not in state or key not in target_state:
            continue
        loaded = state[key]
        target = target_state[key]
        if tuple(loaded.shape) == tuple(target.shape):
            continue
        if loaded.shape[0] < target.shape[0] and loaded.shape[1] == target.shape[1]:
            grown = target.clone()
            torch.nn.init.normal_(grown, std=0.02)
            grown[: loaded.shape[0]] = loaded
            state[key] = grown
            print(
                f"  resized {key}: {tuple(loaded.shape)} -> {tuple(grown.shape)} "
                f"(+{target.shape[0] - loaded.shape[0]} new rows)"
            )
        else:
            raise ValueError(
                f"Cannot resize {key}: loaded {tuple(loaded.shape)} vs "
                f"target {tuple(target.shape)}"
            )


def _load_checkpoint_weights(model, path: str):
    print(f"\nLoading weights from {path}...")
    checkpoint = torch.load(path, weights_only=False)
    state = checkpoint['model_state_dict']
    _resize_embeddings_for_vocab(model, state)
    model.load_state_dict(state)
    return checkpoint


def main():
    args = build_argparser().parse_args()

    if args.stage == 'sft' and args.lr is None:
        args.lr = 5e-5
    elif args.lr is None:
        args.lr = 1e-4

    args.dataset = _resolve_stage_dataset(args.stage, args.dataset)
    token_budget = args.token_budget if args.token_budget > 0 else None

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
    run_tag = f"{args.preset}_{args.stage}_{args.dataset}"
    log_path = log_dir / f'v11_{run_tag}.log'
    tee = TeeLogger(log_path, mode='a' if args.resume else 'w')
    sys.stdout = tee
    print(f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('=' * 60)
    print('  V11: PAM with new memory dynamics')
    print(f"  Preset: {args.preset} | Stage: {args.stage} | Dataset: {args.dataset}")
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
    if token_budget:
        print(f"Token budget: {token_budget:,}")

    seq_len = cfg.max_seq_len
    max_samples = args.max_samples if args.max_samples < 9999999 else None
    print(f"\nLoading {args.dataset} (seq_len={seq_len})...")

    if args.dataset == 'wikitext103':
        train_ds, val_ds, tokenizer = load_wikitext103(max_samples=max_samples, seq_len=seq_len)
    elif args.dataset == 'tinystories':
        train_ds, val_ds, tokenizer = load_tinystories(
            max_samples=max_samples or 20000, seq_len=seq_len)
    elif args.dataset == 'dclm_edu':
        train_ds, val_ds, tokenizer = load_dclm_edu(
            seq_len=seq_len,
            edu_score_min=args.edu_score_min,
            token_budget=token_budget,
        )
    elif args.dataset == 'smoltalk2':
        train_ds, val_ds, tokenizer = load_smoltalk2(
            seq_len=seq_len,
            max_samples=max_samples,
            sft_filter=args.sft_filter,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Match model vocab to the data tokenizer (chat SFT adds ChatML specials -> 50259).
    tok_vocab = len(tokenizer)
    if tok_vocab != cfg.vocab_size:
        print(f"Adjusting vocab_size: {cfg.vocab_size} -> {tok_vocab} (tokenizer)")
        cfg.vocab_size = tok_vocab

    from torch.utils.data import DataLoader
    use_cuda = torch.cuda.is_available()
    is_streaming = args.dataset == 'dclm_edu'
    nw = 0 if is_streaming else (args.num_workers if use_cuda else 0)
    dl_kwargs = {}
    if nw > 0:
        dl_kwargs['persistent_workers'] = True
        dl_kwargs['prefetch_factor'] = 4
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    def _wi(wid):
        np.random.seed(args.seed + wid)
        random.seed(args.seed + wid)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not is_streaming,
        num_workers=nw,
        pin_memory=use_cuda,
        generator=gen,
        worker_init_fn=_wi if nw > 0 else None,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(nw, 2) if nw > 0 else 0,
        pin_memory=use_cuda,
        worker_init_fn=_wi if nw > 0 else None,
    )
    try:
        n_train_batches = len(train_loader)
    except TypeError:
        n_train_batches = None
    print(f"Train batches: {n_train_batches or 'streaming'}, Val batches: {len(val_loader)}")

    model = V11LM(cfg)
    params = model.count_parameters()
    print(f"\nModel parameters: {params}")
    print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")

    start_epoch = 0
    checkpoint = None
    if args.resume:
        checkpoint = _load_checkpoint_weights(model, args.resume)
        start_epoch = checkpoint.get('epoch', 0) + 1
    elif args.resume_from:
        checkpoint = _load_checkpoint_weights(model, args.resume_from)

    if token_budget:
        est_steps = token_budget // max(args.batch_size * seq_len, 1) + args.warmup_steps
        max_epochs = max(args.epochs, 9999)
    else:
        est_steps = None
        max_epochs = args.epochs

    trainer = V7Trainer(
        model, train_loader, val_loader, tokenizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        max_epochs=max_epochs,
        checkpoint_dir=args.checkpoint_dir,
        amp_dtype_str=args.amp_dtype,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        gen_every=args.gen_every,
        gen_prompt=args.gen_prompt,
        log_interval=args.log_interval,
        start_epoch=start_epoch,
        run_label=f'V11/{args.preset}/{args.stage}',
        log_path=str(log_path),
        token_budget=token_budget,
        total_steps_override=est_steps,
    )
    if checkpoint and args.resume and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.global_tokens = checkpoint.get('global_tokens', 0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))

    _summary = [
        f"Host: {os.uname().nodename}",
        f"Preset: {args.preset} | Stage: {args.stage} | Dataset: {args.dataset}",
        f"Memory dynamics: decay_mode={cfg.decay_mode} | write_mode={cfg.write_mode} "
        f"| n_states={cfg.n_states} | chunk_size={cfg.chunk_size}",
        f"Complex dim: {cfg.dim} | Layers: {cfg.n_layers} | "
        f"PAM heads={cfg.n_heads} d={cfg.head_dim} | CGU expand={cfg.expand}",
        f"Activation: {cfg.activation} | RoPE: {cfg.use_rope} | GSP: {cfg.use_gsp} | "
        f"grad_ckpt: {cfg.gradient_checkpointing}",
        f"Params: {params['total']:,} ({params['total']/1e6:.1f}M)",
        f"Epochs: {args.epochs} | Batch: {args.batch_size} | "
        f"Batches/epoch: {n_train_batches or 'stream'} | seq_len={cfg.max_seq_len}",
        f"LR: {args.lr} | warmup={args.warmup_steps} | wd={args.weight_decay} | "
        f"grad_clip={args.gradient_clip} | dropout={cfg.dropout}",
        f"Token budget: {token_budget or 'none'} | edu_score_min: {args.edu_score_min} | "
        f"sft_filter: {args.sft_filter}",
        f"AMP: {args.amp_dtype} | Compile: {args.compile}",
        f"Resume: {args.resume or 'none'} | Weights from: {args.resume_from or 'scratch'}",
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
        _notify_training_failure('stopped by user')
        raise
    except Exception as e:
        _notify_training_failure('OOM' if _is_oom_error(e) else 'failed', e)
        raise
