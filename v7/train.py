"""
V7 training script.

Usage:
    uv run python -m v7.train --preset tiny --epochs 2 --max_samples 100    # smoke
    uv run python -m v7.train --preset medium --epochs 10                   # WikiText-103
    uv run python -m v7.train --preset medium --dataset tinystories         # TinyStories
    uv run python -m v7.train --resume checkpoints_v7/best_model.pt        # resume
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings(
    'ignore',
    message=r'.*Online softmax is disabled.*Inductor.*split the reduction.*',
    category=UserWarning,
    module=r'torch\._inductor\.lowering',
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from v7.model import V7LM, V7Config, get_config
from v7.data import (
    load_wikitext103,
    load_tinystories,
    TeeLogger,
    compute_text_quality,
    resolve_amp_dtype,
    build_lr_scheduler,
    build_param_groups,
)


# ── Discord Notifications ─────────────────────────────────────────────────────

def _notify_discord(content: str) -> None:
    hook = os.environ.get("DISCORD_HOOK", "").strip()
    if not hook:
        return
    if len(content) > 2000:
        content = content[:1997] + "..."
    try:
        payload = json.dumps({"content": content}).encode("utf-8")
        req = urllib.request.Request(
            hook, data=payload, method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "qllm2-v7-notify/1.0",
            },
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[Discord] Webhook send failed: {e}", file=sys.stderr)


def _notify_discord_long(text: str, *, limit: int = 1900) -> None:
    hook = os.environ.get("DISCORD_HOOK", "").strip()
    if not hook:
        return
    lines = text.splitlines(keepends=True)
    chunk: List[str] = []
    chunk_len = 0
    for line in lines:
        if chunk and chunk_len + len(line) > limit:
            _notify_discord("".join(chunk))
            chunk, chunk_len = [], 0
        chunk.append(line)
        chunk_len += len(line)
    if chunk:
        _notify_discord("".join(chunk))


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg


def _notify_training_failure(status: str, exc: Optional[BaseException] = None) -> None:
    parts = [
        f"**V7 Training {status}**",
        f"Host: {os.uname().nodename}",
        f"Command: {' '.join(sys.argv)}",
    ]
    if exc is not None:
        parts.append(f"Error: {type(exc).__name__}: {exc}")
    _notify_discord("\n".join(parts))


# ── Trainer ───────────────────────────────────────────────────────────────────

class V7Trainer:
    def __init__(
        self,
        model: V7LM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        tokenizer,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        gradient_clip: float = 1.0,
        max_epochs: int = 10,
        checkpoint_dir: str = 'checkpoints_v7',
        amp_dtype_str: str = 'auto',
        compile_model: bool = False,
        compile_mode: str = 'default',
        gen_every: int = 0,
        gen_prompt: str = 'The',
        log_interval: int = 50,
        start_epoch: int = 0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.max_epochs = max_epochs
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gen_every = gen_every
        self.gen_prompt = gen_prompt
        self.log_interval = log_interval
        self.start_epoch = start_epoch

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        param_groups = build_param_groups(model, weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=learning_rate, betas=(0.9, 0.95),
        )

        total_steps = max_epochs * len(train_loader)
        self.scheduler = build_lr_scheduler(
            self.optimizer, 'warmup_cosine', warmup_steps, total_steps,
        )

        self.amp_dtype = resolve_amp_dtype(amp_dtype_str)
        self.use_amp = self.amp_dtype is not None
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if self.use_amp and self.amp_dtype == torch.float16
            else None
        )

        if compile_model:
            print(f"Compiling model (mode={compile_mode})...")
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
            except Exception as e:
                print(f"torch.compile failed ({e}), continuing without")

        self.global_step = 0
        self.global_tokens = 0
        self.best_val_loss = float('inf')
        self.best_val_ppl = float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss_w = 0.0
        total_tokens = 0
        epoch_start = time.time()
        log_start = epoch_start
        log_tokens = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            batch_tokens = input_ids.shape[0] * input_ids.shape[1]

            with torch.amp.autocast(
                self.device.type,
                enabled=self.use_amp,
                dtype=self.amp_dtype or torch.float16,
            ):
                logits, _ = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1),
                )

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip,
                )
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1
            self.optimizer.zero_grad(set_to_none=True)

            self.global_tokens += batch_tokens
            total_tokens += batch_tokens
            log_tokens += batch_tokens
            total_loss_w += loss.item() * batch_tokens

            if batch_idx % self.log_interval == 0:
                ppl = math.exp(min(loss.item(), 20))
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                avg_tok_s = total_tokens / elapsed if elapsed > 0 else 0
                int_elapsed = time.time() - log_start
                inst_tok_s = log_tokens / int_elapsed if int_elapsed > 0 else 0

                n_total = len(self.train_loader)
                pct = 100.0 * (batch_idx + 1) / n_total
                remaining = (
                    elapsed / (batch_idx + 1) * (n_total - batch_idx - 1)
                    if batch_idx > 0 else 0
                )
                eta_m, eta_s = divmod(int(remaining), 60)

                line = (
                    f"  [{epoch+1}] {batch_idx}/{n_total} ({pct:.0f}%) "
                    f"loss={loss.item():.4f} ppl={ppl:.1f} lr={lr:.2e} "
                    f"| {inst_tok_s:.0f} tok/s (avg {avg_tok_s:.0f}) "
                    f"ETA {eta_m}m{eta_s:02d}s"
                )
                if self.device.type == 'cuda':
                    mem = torch.cuda.memory_allocated() / 1e9
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    line += f" | GPU {mem:.1f}/{peak:.1f}GB"
                line += f" | gtok={self.global_tokens}"
                print(line)

                log_start = time.time()
                log_tokens = 0

            if (
                self.gen_every > 0
                and batch_idx > 0
                and batch_idx % self.gen_every == 0
                and self.tokenizer is not None
            ):
                try:
                    text = self._generate_sample(self.gen_prompt)
                    print(
                        f"  [mid-epoch sample @ batch {batch_idx}, "
                        f"{self.global_tokens:,} tok]"
                    )
                    print(f"  Prompt: {self.gen_prompt}")
                    print(f"  Generated: {text}")
                    ppl = math.exp(min(loss.item(), 20))
                    lr = self.scheduler.get_last_lr()[0]
                    _gen_msg = (
                        f"**[V7 gen_every]** Epoch {epoch+1} batch {batch_idx} "
                        f"({self.global_tokens:,} tok)\n"
                        f"loss={loss.item():.4f} ppl={ppl:.1f} lr={lr:.2e} | "
                        f"{avg_tok_s:.0f} tok/s\n"
                        f"Prompt: {self.gen_prompt}\n"
                        f"Generated: {(text[:800] + '...') if len(text) > 800 else text}"
                    )
                    _notify_discord(_gen_msg)
                except Exception:
                    pass
                self.model.train()

        epoch_elapsed = time.time() - epoch_start
        avg_tok_s = total_tokens / epoch_elapsed if epoch_elapsed > 0 else 0
        avg_loss = total_loss_w / max(total_tokens, 1)
        return {
            'loss': avg_loss,
            'ppl': math.exp(min(avg_loss, 20)),
            'avg_tok_s': avg_tok_s,
            'epoch_tokens': total_tokens,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}
        self.model.eval()
        total_loss_w = 0.0
        total_tokens = 0
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            batch_tokens = input_ids.shape[0] * input_ids.shape[1]

            with torch.amp.autocast(
                self.device.type,
                enabled=self.use_amp,
                dtype=self.amp_dtype or torch.float16,
            ):
                logits, _ = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1),
                )

            total_loss_w += loss.item() * batch_tokens
            total_tokens += batch_tokens

        if total_tokens == 0:
            return {}
        avg_loss = total_loss_w / total_tokens
        return {'val_loss': avg_loss, 'val_ppl': math.exp(min(avg_loss, 20))}

    @torch.no_grad()
    def _generate_sample(self, prompt: str = "The", max_tokens: int = 100) -> str:
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        generated = m.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        return self.tokenizer.decode(generated[0].tolist())

    def save_checkpoint(self, name: str, epoch: int):
        path = self.checkpoint_dir / name
        m = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        ckpt = {
            'model_state_dict': m.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'global_tokens': self.global_tokens,
            'best_val_loss': self.best_val_loss,
            'best_val_ppl': self.best_val_ppl,
            'epoch': epoch,
            'config': asdict(m.config),
        }
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")

    def train(self):
        training_start = time.time()
        print(f"\nTraining on {self.device}")
        m = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        params = m.count_parameters()
        print(f"Parameters: {params}")
        print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")
        print(
            f"Epochs: {self.start_epoch+1}..{self.max_epochs}, "
            f"Batches/epoch: {len(self.train_loader)}"
        )
        print()

        for epoch in range(self.start_epoch, self.max_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.max_epochs}")
            print('=' * 60)

            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - training_start

            line = (
                f"Epoch {epoch+1}/{self.max_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"PPL: {train_metrics['ppl']:.2f} | "
                f"{train_metrics['avg_tok_s']:.0f} tok/s | "
                f"Time: {epoch_time:.1f}s ({self.global_tokens:,} tok)"
            )

            is_best = False
            if self.val_loader is not None and len(self.val_loader) > 0:
                val_metrics = self.validate()
                line += (
                    f" | Val Loss: {val_metrics['val_loss']:.4f} "
                    f"PPL: {val_metrics['val_ppl']:.2f}"
                )
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.best_val_ppl = val_metrics['val_ppl']
                    line += " *best*"
                    is_best = True
            print(line)

            if is_best:
                self.save_checkpoint('best_model.pt', epoch)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch)

            epoch_text = ""
            if self.tokenizer is not None:
                try:
                    text = self._generate_sample(self.gen_prompt)
                    epoch_text = text
                    print(f"\nPrompt: {self.gen_prompt}")
                    print(f"Generated: {text}")
                    qm = compute_text_quality(text)
                    print(
                        f"  Quality: rep3={qm['repeat_3gram']:.3f} "
                        f"rep4={qm['repeat_4gram']:.3f} "
                        f"restarts={qm['restart_frag']:.0f} "
                        f"uniq={qm['unique_word_ratio']:.3f}"
                    )
                except Exception as e:
                    print(f"(Sample generation failed: {e})")

            _ep_msg = (
                f"**V7 Epoch {epoch+1}/{self.max_epochs}**\n"
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"PPL: {train_metrics['ppl']:.2f} | "
                f"{train_metrics['avg_tok_s']:.0f} tok/s | "
                f"Time: {epoch_time:.1f}s"
            )
            if self.val_loader is not None and len(self.val_loader) > 0:
                _ep_msg += (
                    f"\nVal Loss: {val_metrics['val_loss']:.4f} "
                    f"PPL: {val_metrics['val_ppl']:.2f}"
                )
                if is_best:
                    _ep_msg += " *best*"
            if epoch_text:
                _ep_msg += (
                    f"\nPrompt: {self.gen_prompt}\nGenerated: "
                    f"{(epoch_text[:600] + '...') if len(epoch_text) > 600 else epoch_text}"
                )
            _notify_discord(_ep_msg)

        self.save_checkpoint('final_model.pt', self.max_epochs - 1)

        total_time = time.time() - training_start
        _done_msg = (
            f"**V7 Training complete!**\n"
            f"Wall time: {total_time:.1f}s ({total_time/3600:.2f}h)\n"
            f"Best Val Loss: {self.best_val_loss:.4f}, "
            f"Best Val PPL: {self.best_val_ppl:.2f}"
        )
        print(f"\nTraining complete!")
        print(f"Total wall time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        print(
            f"Best Val Loss: {self.best_val_loss:.4f}, "
            f"Best Val PPL: {self.best_val_ppl:.2f}"
        )
        _notify_discord(_done_msg)


def main():
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        try:
            for line in _env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip().strip("'\"")
                    if key == "DISCORD_HOOK" and value:
                        os.environ["DISCORD_HOOK"] = value
                        break
        except Exception as _e:
            print(f"[Discord] Could not read .env: {_e}", file=sys.stderr)
    if os.environ.get("DISCORD_HOOK"):
        print("[Discord] Webhook configured — notifications enabled", file=sys.stderr)
    else:
        print("[Discord] No webhook (set DISCORD_HOOK in .env to enable)", file=sys.stderr)

    parser = argparse.ArgumentParser(description='V7 PAM Language Model Training')
    parser.add_argument(
        '--preset', type=str, default='medium_h6',
        choices=['tiny', 'medium', 'medium_h6'],
    )
    parser.add_argument(
        '--dataset', type=str, default='wikitext103',
        choices=['wikitext103', 'tinystories'],
    )
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--seq_len', type=int, default=None,
                        help='Override preset seq_len')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override preset dropout')
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--amp_dtype', type=str, default='auto',
                        choices=['auto', 'bf16', 'fp16'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=9999999)
    parser.add_argument('--gen_every', type=int, default=5000)
    parser.add_argument('--gen_prompt', type=str,
                        default='In 1923 , the University of')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v7')
    parser.add_argument('--resume', type=str, default=None)

    # Ablation toggles
    parser.add_argument('--no_rope', action='store_true')
    parser.add_argument('--no_gsp', action='store_true')
    parser.add_argument('--no_fused_qkv', action='store_true')
    parser.add_argument('--qk_norm', action='store_true')
    parser.add_argument('--no_hierarchical_dt', action='store_true',
                        help='Disable hierarchical timescale (uniform dt_bias=-4.0)')

    args = parser.parse_args()

    # Logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'v7_{args.preset}_{args.dataset}.log'
    log_mode = 'a' if args.resume else 'w'
    tee = TeeLogger(log_path, mode=log_mode)
    sys.stdout = tee

    print(f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("  V7: Phase-Associative Memory Language Model")
    print(f"  Preset: {args.preset} | Dataset: {args.dataset}")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Config
    cfg = get_config(args.preset)
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

    print(f"\nConfig: {asdict(cfg)}")
    print(
        f"Training: lr={args.lr}, warmup={args.warmup_steps}, wd={args.weight_decay}, "
        f"grad_clip={args.gradient_clip}"
    )
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"AMP: {args.amp_dtype}, Compile: {args.compile}")

    # Data
    seq_len = cfg.max_seq_len
    max_samples = args.max_samples if args.max_samples < 9999999 else None
    print(f"\nLoading {args.dataset} (seq_len={seq_len})...")

    if args.dataset == 'wikitext103':
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
        dl_kwargs['persistent_workers'] = True
        dl_kwargs['prefetch_factor'] = 4

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=use_cuda, **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=use_cuda, **dl_kwargs,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = V7LM(cfg)
    params = model.count_parameters()
    print(f"\nModel parameters: {params}")
    print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")

    # Resume
    start_epoch = 0
    checkpoint = None
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Trainer
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
    )

    if checkpoint and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.global_tokens = checkpoint.get('global_tokens', 0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))

    _summary_lines = [
        f"Preset: {args.preset} | Dataset: {args.dataset}",
        f"Config: dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} "
        f"head_dim={cfg.head_dim} expand={cfg.expand}",
        f"hierarchical_dt={cfg.hierarchical_dt}",
        f"Params: {params['total']:,} ({params['total']/1e6:.1f}M)",
        f"seq_len={cfg.max_seq_len} batch_size={args.batch_size} epochs={args.epochs}",
        f"lr={args.lr} warmup={args.warmup_steps} wd={args.weight_decay}",
        f"AMP: {args.amp_dtype} Compile: {args.compile}",
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}",
        f"Host: {os.uname().nodename}",
    ]
    _discord_header = "**V7 Training started**" if not args.resume else "**V7 Training resumed**"
    _notify_discord_long(
        _discord_header + "\n```\n" + "\n".join(_summary_lines) + "\n```"
    )

    trainer.train()

    print(f"\nWall clock end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee._stdout
    tee.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        _notify_training_failure("stopped by user")
        raise
    except Exception as e:
        _notify_training_failure("OOM" if _is_oom_error(e) else "failed", e)
        raise
