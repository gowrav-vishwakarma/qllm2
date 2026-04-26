"""
V9 training script.

Reuses V7's data pipeline and trainer, but instantiates V9LM so logs,
checkpoints, and presets stay separate from the V7 baseline.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from v7.data import (  # noqa: E402
    TeeLogger,
    load_tinystories,
    load_wikitext103,
)
from v7.train import (  # noqa: E402
    V7Trainer,
    _is_oom_error,
    _notify_discord_long,
    _notify_training_failure,
    seed_everything,
)
from v9.model import PRESETS, V9LM, get_config  # noqa: E402


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V9 PAM-upgrade model")
    parser.add_argument("--preset", type=str, default="medium_h16_flat", choices=list(PRESETS.keys()))
    parser.add_argument("--dataset", type=str, default="wikitext103", choices=["wikitext103", "tinystories"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="default")
    parser.add_argument("--amp_dtype", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=9999999)
    parser.add_argument("--gen_every", type=int, default=5000)
    parser.add_argument("--gen_prompt", type=str, default="In 1923 , the University of")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_v9")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--activation", type=str, default=None, choices=["modrelu", "swish", "phase_mod"])
    parser.add_argument("--no_rope", action="store_true")
    parser.add_argument("--no_gsp", action="store_true")
    parser.add_argument("--no_fused_qkv", action="store_true")
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument("--no_hierarchical_dt", action="store_true")
    parser.add_argument("--no_cross_level", action="store_true")
    parser.add_argument("--no_grad_ckpt", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--unitary_lambda", type=float, default=0.0)
    parser.add_argument("--multi_scale_loss", action="store_true")
    parser.add_argument("--aux_loss_weight", type=float, default=0.1)
    parser.add_argument("--aux_layer_stride", type=int, default=3)
    parser.add_argument("--max_aux_offset", type=int, default=32)
    parser.add_argument("--no_reverse_assoc", action="store_true")

    parser.add_argument("--pam_output_gate", action="store_true")
    parser.add_argument("--pam_short_conv", type=int, default=None)
    return parser


def _apply_overrides(cfg, args):
    if args.seq_len is not None:
        cfg.max_seq_len = args.seq_len
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.no_rope:
        cfg.use_rope = False
    if args.no_gsp:
        cfg.use_gsp = False
    if args.no_fused_qkv:
        cfg.fused_qkv = False
    if args.qk_norm:
        cfg.qk_norm = True
    if args.no_hierarchical_dt:
        cfg.hierarchical_dt = False
        cfg.dt_bias_schedule = None
    if args.no_cross_level:
        cfg.cross_level = False
    if args.no_grad_ckpt:
        cfg.gradient_checkpointing = False
    if args.activation is not None:
        cfg.activation = args.activation
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    if args.multi_scale_loss:
        cfg.multi_scale_loss = True
        cfg.aux_loss_weight = args.aux_loss_weight
        cfg.aux_layer_stride = args.aux_layer_stride
        cfg.max_aux_offset = args.max_aux_offset
    if args.no_reverse_assoc:
        cfg.use_reverse_assoc = False
    if args.pam_output_gate:
        cfg.pam_output_gate = True
    if args.pam_short_conv is not None:
        cfg.pam_short_conv = args.pam_short_conv
    return cfg


def main() -> None:
    _load_dotenv()
    args = build_parser().parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"v9_{args.preset}_{args.dataset}.log"
    log_mode = "a" if args.resume else "w"
    tee = TeeLogger(log_path, mode=log_mode)
    sys.stdout = tee

    print(f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("  V9: PAM Upgrade Language Model")
    print(f"  Preset: {args.preset} | Dataset: {args.dataset}")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed_everything(args.seed)
    print(f"Seed: {args.seed}")

    cfg = _apply_overrides(get_config(args.preset), args)
    print(f"\nConfig: {asdict(cfg)}")
    print(
        f"Training: lr={args.lr}, warmup={args.warmup_steps}, wd={args.weight_decay}, "
        f"grad_clip={args.gradient_clip}"
    )
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"AMP: {args.amp_dtype}, Compile: {args.compile}")

    seq_len = cfg.max_seq_len
    max_samples = args.max_samples if args.max_samples < 9999999 else None
    print(f"\nLoading {args.dataset} (seq_len={seq_len})...")
    if args.dataset == "wikitext103":
        train_ds, val_ds, tokenizer = load_wikitext103(
            max_samples=max_samples, seq_len=seq_len,
        )
    else:
        train_ds, val_ds, tokenizer = load_tinystories(
            max_samples=max_samples or 20000, seq_len=seq_len,
        )

    use_cuda = torch.cuda.is_available()
    nw = args.num_workers if use_cuda else 0
    dl_kwargs = {}
    if nw > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4

    dl_generator = torch.Generator()
    dl_generator.manual_seed(args.seed)

    def _worker_init_fn(worker_id: int) -> None:
        np.random.seed(args.seed + worker_id)
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=use_cuda,
        generator=dl_generator, worker_init_fn=_worker_init_fn,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=use_cuda,
        worker_init_fn=_worker_init_fn, **dl_kwargs,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = V9LM(cfg)
    params = model.count_parameters()
    print(f"\nModel parameters: {params}")
    print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")

    start_epoch = 0
    checkpoint = None
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    trainer = V7Trainer(
        model, train_loader, val_loader, tokenizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        max_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        amp_dtype_str=args.amp_dtype,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        gen_every=args.gen_every,
        gen_prompt=args.gen_prompt,
        log_interval=args.log_interval,
        start_epoch=start_epoch,
        unitary_lambda=args.unitary_lambda,
        run_label="V9",
        log_path=str(log_path),
        log_dir=str(log_dir),
    )

    if checkpoint and "optimizer_state_dict" in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.global_step = checkpoint.get("global_step", 0)
        trainer.global_tokens = checkpoint.get("global_tokens", 0)
        trainer.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        trainer.best_val_ppl = checkpoint.get("best_val_ppl", float("inf"))

    summary_lines = [
        f"Model: V9 | Preset: {args.preset} | Dataset: {args.dataset}",
        f"Config: dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} "
        f"head_dim={cfg.head_dim} expand={cfg.expand}",
        f"V9 flags: pam_output_gate={cfg.pam_output_gate} pam_short_conv={cfg.pam_short_conv}",
        f"Params: {params['total']:,} ({params['total']/1e6:.1f}M)",
        f"seq_len={cfg.max_seq_len} batch_size={args.batch_size} epochs={args.epochs}",
        f"lr={args.lr} warmup={args.warmup_steps} wd={args.weight_decay}",
        f"AMP: {args.amp_dtype} Compile: {args.compile}",
        f"Log dir: {log_dir}",
        f"Log file: {log_path}",
        f"Checkpoint dir: {args.checkpoint_dir}",
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}",
        f"Host: {os.uname().nodename}",
    ]
    header = "**V9 Training started**" if not args.resume else "**V9 Training resumed**"
    _notify_discord_long(header + "\n```\n" + "\n".join(summary_lines) + "\n```")

    trainer.train()

    print(f"\nWall clock end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _notify_training_failure("stopped by user")
        raise
    except Exception as e:
        _notify_training_failure("OOM" if _is_oom_error(e) else "failed", e)
        raise
