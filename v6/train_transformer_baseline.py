"""
Train a standard transformer baseline on WikiText-103.

Reuses the exact same data pipeline (GPT-2 tokenizer, same preprocessing,
same TextDataset chunking) and evaluation loop (token-weighted cross-entropy
-> perplexity) as v6/train.py, for an apples-to-apples comparison against
PAM models.

Usage:
    python -m v6.train_transformer_baseline                     # defaults
    python -m v6.train_transformer_baseline --epochs 3          # quick test
    python -m v6.train_transformer_baseline --batch_size 2      # if OOM
    python -m v6.train_transformer_baseline --resume CKPT_PATH  # resume
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
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

from v6.train import (
    load_wikitext103,
    TextDataset,
    TeeLogger,
    compute_text_quality,
    _resolve_amp_dtype,
    _build_lr_scheduler,
)
from v6.transformer_baseline import TransformerLM, get_transformer_config_100m


def _build_param_groups(model: nn.Module, weight_decay: float):
    """Split parameters into decay / no-decay groups for AdamW.

    Standard transformer convention: decay 2-D weight matrices,
    do not decay biases, embeddings, or LayerNorm parameters.
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:
            no_decay_params.append(param)
        elif 'embed' in name:
            no_decay_params.append(param)
        elif 'ln' in name or 'norm' in name:
            no_decay_params.append(param)
        elif 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Param groups: {len(decay_params)} tensors ({n_decay:,} params) with weight decay, "
          f"{len(no_decay_params)} tensors ({n_no_decay:,} params) without")

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]


class TransformerTrainer:
    """Training loop for the transformer baseline.

    Mirrors v6 Trainer's evaluation loop exactly: token-weighted
    cross-entropy loss, perplexity = exp(avg_loss).
    """

    def __init__(
        self,
        model: TransformerLM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        tokenizer,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        gradient_clip: float = 1.0,
        max_epochs: int = 10,
        checkpoint_dir: str = 'checkpoints_transformer_baseline',
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

        param_groups = _build_param_groups(model, weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
        )

        total_steps = max_epochs * len(train_loader)
        self.scheduler = _build_lr_scheduler(
            self.optimizer, 'warmup_cosine', warmup_steps, total_steps,
        )

        self.amp_dtype = _resolve_amp_dtype(amp_dtype_str)
        self.use_amp = self.amp_dtype is not None
        self.scaler = (
            torch.amp.GradScaler('cuda')
            if self.use_amp and self.amp_dtype == torch.float16
            else None
        )

        if compile_model:
            print(f"Compiling model with torch.compile (mode={compile_mode})...")
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
            except Exception as e:
                print(f"torch.compile failed ({e}), continuing without compilation")

        self.global_step = 0
        self.global_tokens = 0
        self.best_val_loss = float('inf')
        self.best_val_ppl = float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss_w = 0.0
        total_tokens = 0
        epoch_start = time.time()
        log_interval_start = epoch_start
        log_interval_tokens = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            batch_tokens = input_ids.shape[0] * input_ids.shape[1]

            with torch.amp.autocast(self.device.type, enabled=self.use_amp,
                                    dtype=self.amp_dtype or torch.float16):
                logits = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1
            self.optimizer.zero_grad(set_to_none=True)

            self.global_tokens += batch_tokens
            total_tokens += batch_tokens
            log_interval_tokens += batch_tokens
            total_loss_w += loss.item() * batch_tokens

            if batch_idx % self.log_interval == 0:
                ppl = math.exp(min(loss.item(), 20))
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                avg_tok_s = total_tokens / elapsed if elapsed > 0 else 0
                interval_elapsed = time.time() - log_interval_start
                inst_tok_s = log_interval_tokens / interval_elapsed if interval_elapsed > 0 else 0

                n_total = len(self.train_loader)
                pct = 100.0 * (batch_idx + 1) / n_total
                remaining = elapsed / (batch_idx + 1) * (n_total - batch_idx - 1) if batch_idx > 0 else 0
                eta_m, eta_s = divmod(int(remaining), 60)

                line = (
                    f"  [{epoch+1}] {batch_idx}/{n_total} ({pct:.0f}%) "
                    f"loss={loss.item():.4f} ppl={ppl:.1f} lr={lr:.2e} "
                    f"| {inst_tok_s:.0f} tok/s (avg {avg_tok_s:.0f}) "
                    f"ETA {eta_m}m{eta_s:02d}s"
                )
                if self.device.type == 'cuda':
                    mem = torch.cuda.memory_allocated() / 1e9
                    mem_res = torch.cuda.max_memory_allocated() / 1e9
                    line += f" | GPU {mem:.1f}/{mem_res:.1f}GB"
                line += f" | gtok={self.global_tokens}"
                print(line)

                log_interval_start = time.time()
                log_interval_tokens = 0

            # Mid-epoch generation
            if (self.gen_every > 0 and batch_idx > 0
                    and batch_idx % self.gen_every == 0
                    and self.tokenizer is not None):
                try:
                    text = self._generate_sample(self.gen_prompt)
                    print(f"  [mid-epoch sample @ batch {batch_idx}, {self.global_tokens:,} tok]")
                    print(f"  Prompt: {self.gen_prompt}")
                    print(f"  Generated: {text}")
                except Exception:
                    pass
                self.model.train()

        epoch_elapsed = time.time() - epoch_start
        avg_tok_s = total_tokens / epoch_elapsed if epoch_elapsed > 0 else 0
        safe_tokens = max(total_tokens, 1)
        avg_loss = total_loss_w / safe_tokens
        return {
            'loss': avg_loss,
            'ppl': math.exp(min(avg_loss, 20)),
            'avg_tok_s': avg_tok_s,
            'epoch_tokens': total_tokens,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop -- identical to v6 Trainer.validate().

        Token-weighted cross-entropy averaged over all validation tokens,
        perplexity = exp(avg_loss).
        """
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}
        self.model.eval()
        total_loss_w = 0.0
        total_tokens = 0
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            batch_tokens = input_ids.shape[0] * input_ids.shape[1]

            logits = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
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
        model_to_gen = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        generated = model_to_gen.generate(
            prompt_tensor, max_new_tokens=max_tokens, temperature=0.8,
            top_k=50, top_p=0.9, repetition_penalty=1.2,
        )
        return self.tokenizer.decode(generated[0].tolist())

    def save_checkpoint(self, name: str, epoch: int):
        path = self.checkpoint_dir / name
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        ckpt = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'global_tokens': self.global_tokens,
            'best_val_loss': self.best_val_loss,
            'best_val_ppl': self.best_val_ppl,
            'epoch': epoch,
            'config': {
                'vocab_size': model_to_save.config.vocab_size,
                'max_seq_len': model_to_save.config.max_seq_len,
                'd_model': model_to_save.config.d_model,
                'n_layers': model_to_save.config.n_layers,
                'n_heads': model_to_save.config.n_heads,
                'd_ff': model_to_save.config.d_ff,
                'dropout': model_to_save.config.dropout,
                'tie_weights': model_to_save.config.tie_weights,
            },
        }
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")

    def train(self):
        training_start = time.time()
        print(f"\nTraining on {self.device}")
        params = self.model.count_parameters() if hasattr(self.model, 'count_parameters') else {}
        model_for_params = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        if hasattr(model_for_params, 'count_parameters'):
            params = model_for_params.count_parameters()
        if params:
            print(f"Parameters: {params}")
            print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")
        print(f"Epochs: {self.start_epoch+1}..{self.max_epochs}, "
              f"Batches/epoch: {len(self.train_loader)}")
        print()

        for epoch in range(self.start_epoch, self.max_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.max_epochs}")
            print('=' * 60)

            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

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

            # End-of-epoch generation
            if self.tokenizer is not None:
                try:
                    text = self._generate_sample(self.gen_prompt)
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

        self.save_checkpoint('final_model.pt', self.max_epochs - 1)

        total_time = time.time() - training_start
        print(f"\nTraining complete!")
        print(f"Total wall time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        print(f"Best Val Loss: {self.best_val_loss:.4f}, Best Val PPL: {self.best_val_ppl:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train transformer baseline (~100M) on WikiText-103'
    )
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--max_samples', type=int, default=9999999,
                        help='Max training samples (9999999 = full dataset)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--amp_dtype', type=str, default='auto',
                        choices=['auto', 'bf16', 'fp16'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gen_every', type=int, default=5000)
    parser.add_argument('--gen_prompt', type=str, default='In 1923 , the University of')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_transformer_baseline')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Set up logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'transformer_baseline.log'
    log_mode = 'a' if args.resume else 'w'
    tee = TeeLogger(log_path, mode=log_mode)
    sys.stdout = tee

    print(f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("  Transformer Baseline (~100M params)")
    print("  GPT-2-style decoder-only transformer")
    print("  Same data pipeline as PAM v3 for fair comparison")
    print("=" * 60)

    # CUDA performance settings
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Model config
    config = get_transformer_config_100m()
    config.max_seq_len = args.seq_len
    config.dropout = args.dropout

    print(f"Architecture: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, d_ff={config.d_ff}")
    print(f"seq_len={config.max_seq_len}, dropout={config.dropout}, tie_weights={config.tie_weights}")
    print(f"Training: lr={args.lr}, warmup={args.warmup_steps}, wd={args.weight_decay}, "
          f"grad_clip={args.gradient_clip}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"AMP: {args.amp_dtype}, Compile: {args.compile}")

    # Load data -- exact same pipeline as PAM v3
    print("\nLoading WikiText-103 (same pipeline as PAM v3)...")
    max_samples = args.max_samples if args.max_samples < 9999999 else None
    train_ds, val_ds, tokenizer = load_wikitext103(
        max_samples=max_samples,
        seq_len=args.seq_len,
        use_cache=True,
    )

    use_cuda = torch.cuda.is_available()
    nw = args.num_workers if use_cuda else 0
    pm = use_cuda
    dl_kwargs = {}
    if nw > 0:
        dl_kwargs['persistent_workers'] = True
        dl_kwargs['prefetch_factor'] = 4
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw, pin_memory=pm,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw, pin_memory=pm,
        **dl_kwargs,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = TransformerLM(config)
    params = model.count_parameters()
    print(f"\nModel parameters: {params}")
    print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")

    # Handle resume
    start_epoch = 0
    checkpoint = None
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Create trainer
    trainer = TransformerTrainer(
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

    # Restore optimizer/scheduler state on resume
    if checkpoint and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.global_tokens = checkpoint.get('global_tokens', 0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))

    trainer.train()

    print(f"\nWall clock end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee._stdout
    tee.close()


if __name__ == '__main__':
    main()
