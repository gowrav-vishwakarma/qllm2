"""
V6 Training Script.

Usage:
    python -m v6.train --size small-matched --epochs 10 --max_samples 100000
    python -m v6.train --size tiny --epochs 2 --max_samples 100  # smoke test
    python -m v6.train --no_working_memory  # ablation without working memory
    python -m v6.train --resume checkpoints_v6/best_model.pt
"""

import os
import sys
import time
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from v6.model import PhaseFieldLM, create_model, ModelOutput
from v6.config import V6Config, get_config
from v6.init import list_strategies


class TeeLogger:
    """Writes to both stdout and a log file with timestamps."""

    def __init__(self, log_path: Path, mode: str = 'w'):
        self._stdout = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, mode, buffering=1)
        self._at_line_start = True

    def write(self, text: str):
        self._stdout.write(text)
        for char in text:
            if self._at_line_start and char != '\n':
                ts = datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ')
                self._file.write(ts)
                self._at_line_start = False
            self._file.write(char)
            if char == '\n':
                self._at_line_start = True
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    @property
    def encoding(self):
        return self._stdout.encoding

    def fileno(self):
        return self._stdout.fileno()

    def isatty(self):
        return self._stdout.isatty()


class TextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int = 512):
        self.seq_len = seq_len
        n_chunks = len(tokens) // (seq_len + 1)
        self.data = tokens[:n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {'input_ids': chunk[:-1], 'labels': chunk[1:]}


def load_tinystories(max_samples=20000, seq_len=512):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    try:
        from datasets import load_dataset
        print(f"Loading TinyStories (max_samples={max_samples})...")
        ds = load_dataset('roneneldan/TinyStories', split='train')
        texts = [item['text'] for item in ds if item['text'].strip()]
        if max_samples:
            texts = texts[:max_samples]
    except Exception as e:
        print(f"Failed to load TinyStories: {e}")
        print("Using random data as fallback.")
        return _random_dataset(50257, seq_len, 1000), _random_dataset(50257, seq_len, 100), tokenizer

    print(f"Tokenizing {len(texts)} texts...")
    all_tokens = []
    for text in texts:
        toks = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(toks)
        all_tokens.append(tokenizer.eos_token_id)

    all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    print(f"Total tokens: {len(all_tokens):,}")

    split = int(len(all_tokens) * 0.9)
    train_ds = TextDataset(all_tokens[:split], seq_len)
    val_ds = TextDataset(all_tokens[split:], seq_len)
    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
    return train_ds, val_ds, tokenizer


def _random_dataset(vocab_size, seq_len, num_samples):
    tokens = torch.randint(1, vocab_size, (num_samples * (seq_len + 1),))
    return TextDataset(tokens, seq_len)


class Trainer:
    def __init__(
        self,
        model: PhaseFieldLM,
        config: V6Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer=None,
        checkpoint_dir: str = 'checkpoints_v6',
        start_epoch: int = 0,
        save_checkpoints: bool = True,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = start_epoch
        self.save_checkpoints = save_checkpoints
        self.verbose = verbose
        self.gen_every = 0
        self.gen_prompt = "Once upon a time"

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        total_steps = config.max_epochs * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps,
        )

        self.use_amp = self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_ppl = float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_div_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        first_step_start = None

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx == 0:
                first_step_start = time.time()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            seq_len = input_ids.shape[1]

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                output = self.model(input_ids)

                logits = output.logits.view(-1, output.logits.size(-1))
                ce_loss = F.cross_entropy(logits, labels.view(-1))

                loss = ce_loss
                div_loss_val = 0.0
                if output.diversity_loss is not None:
                    total_steps = self.config.max_epochs * len(self.train_loader)
                    progress = min(self.global_step / max(total_steps, 1), 1.0)
                    div_w = self.config.diversity_loss_weight + (
                        self.config.diversity_loss_floor - self.config.diversity_loss_weight
                    ) * progress
                    div_w = max(div_w, self.config.diversity_loss_floor)
                    div_loss = output.diversity_loss * div_w
                    loss = loss + div_loss
                    div_loss_val = div_loss.item()

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_div_loss += div_loss_val
            num_batches += 1

            if self.verbose and batch_idx == 0 and first_step_start is not None:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                first_step_s = time.time() - first_step_start
                print(f"  First step wall time: {first_step_s:.1f}s")

            if self.verbose and batch_idx % 50 == 0:
                ppl = math.exp(min(ce_loss.item(), 20))
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                samples_per_sec = (batch_idx + 1) * self.config.batch_size / elapsed
                tokens_per_sec = samples_per_sec * seq_len
                print(
                    f"  [{epoch+1}] batch {batch_idx}/{len(self.train_loader)} "
                    f"loss={ce_loss.item():.4f} ppl={ppl:.1f} "
                    f"div={div_loss_val:.4f} lr={lr:.2e} "
                    f"| {samples_per_sec:.1f} samples/s | {tokens_per_sec:.0f} tok/s"
                )

            if (self.gen_every > 0 and batch_idx > 0
                    and batch_idx % self.gen_every == 0
                    and self.tokenizer is not None):
                try:
                    text = self.generate_sample(self.gen_prompt)
                    print(f"  [mid-epoch sample @ batch {batch_idx}]")
                    print(f"  Prompt: {self.gen_prompt}")
                    print(f"  Generated: {text}")
                except Exception:
                    pass
                self.model.train()

        return {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'div_loss': total_div_loss / num_batches,
            'ppl': math.exp(min(total_ce_loss / num_batches, 20)),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            output = self.model(input_ids)
            logits = output.logits.view(-1, output.logits.size(-1))
            loss = F.cross_entropy(logits, labels.view(-1))
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss, 'val_ppl': math.exp(min(avg_loss, 20))}

    @torch.no_grad()
    def generate_sample(self, prompt="The quick brown", max_tokens=100):
        self.model.eval()
        model_to_gen = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        generated = model_to_gen.generate(
            prompt_tensor, max_new_tokens=max_tokens, temperature=0.8,
            top_k=50, top_p=0.9, repetition_penalty=1.2,
        )
        return self.tokenizer.decode(generated[0].tolist())

    def save_checkpoint(self, name: str):
        path = self.checkpoint_dir / name
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        ckpt = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'best_val_ppl': self.best_val_ppl,
            'epoch': self._current_epoch,
            'config': self.config.to_dict(),
        }
        if hasattr(self.model, 'initializer_info'):
            ckpt['init_strategy'] = self.model.initializer_info['init_strategy']
            ckpt['init_seed'] = self.model.initializer_info['init_seed']
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")

    def train(self):
        training_start = time.time()
        print(f"\nTraining on {self.device}")
        params = self.model.count_parameters() if hasattr(self.model, 'count_parameters') else {}
        if params:
            print(f"Parameters: {params}")
            print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")
        print(f"Epochs: {self.start_epoch+1}..{self.config.max_epochs}, Batches/epoch: {len(self.train_loader)}")
        print()

        for epoch in range(self.start_epoch, self.config.max_epochs):
            self._current_epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.max_epochs}")
            print('=' * 60)

            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            line = (
                f"Epoch {epoch+1}/{self.config.max_epochs} | "
                f"Train Loss: {train_metrics['ce_loss']:.4f} "
                f"PPL: {train_metrics['ppl']:.2f} | "
                f"Time: {epoch_time:.1f}s"
            )

            is_best = False
            if self.val_loader is not None:
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

            if self.save_checkpoints and is_best:
                self.save_checkpoint('best_model.pt')
            if self.save_checkpoints and (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

            if self.tokenizer is not None:
                try:
                    text = self.generate_sample(self.gen_prompt)
                    print(f"\nPrompt: {self.gen_prompt}")
                    print(f"Generated: {text}")
                except Exception as e:
                    print(f"(Sample generation failed: {e})")

        self._current_epoch = self.config.max_epochs - 1
        if self.save_checkpoints:
            self.save_checkpoint('final_model.pt')

        total_time = time.time() - training_start
        print(f"\nTraining complete!")
        print(f"Total wall time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        print(f"Best Val Loss: {self.best_val_loss:.4f}, Best Val PPL: {self.best_val_ppl:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Train V6 Phase-First LM')
    parser.add_argument('--size', type=str, default='small-matched',
                        choices=['tiny', 'small', 'small-matched', 'medium'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max_samples', type=int, default=20000)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--no_working_memory', action='store_true',
                        help='Disable working memory (ablation)')
    parser.add_argument('--no_internal_memory', action='store_true',
                        help='Disable internal memory (ablation)')
    parser.add_argument('--wm_slots', type=int, default=None,
                        help='Override number of working memory slots')
    parser.add_argument('--im_slots', type=int, default=None,
                        help='Override number of internal memory slots')
    parser.add_argument('--init_strategy', type=str, default=None,
                        choices=list_strategies())
    parser.add_argument('--init_seed', type=int, default=None)
    parser.add_argument('--use_attention', action='store_true',
                        help='Enable PhaseAttention layers (disabled by default)')
    parser.add_argument('--attn_every', type=int, default=0,
                        help='Place attention every N layers (0 = last layer only)')
    parser.add_argument('--gen_every', type=int, default=0,
                        help='Generate a sample every N batches during training (0 = end of epoch only)')
    parser.add_argument('--gen_prompt', type=str, default='Once upon a time',
                        help='Prompt for mid-epoch and end-of-epoch text generation')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v6')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    config = get_config(args.size)
    config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.no_working_memory:
        config.num_wm_slots = 0
    if args.no_internal_memory:
        config.num_im_slots = 0
    if args.wm_slots is not None:
        config.num_wm_slots = args.wm_slots
    if args.im_slots is not None:
        config.num_im_slots = args.im_slots
    if args.init_strategy is not None:
        config.init_strategy = args.init_strategy
    config.init_seed = args.init_seed
    if args.use_attention:
        config.use_attention = True
        config.attn_every = args.attn_every

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'v6_train_{args.size}.log'
    log_mode = 'a' if args.resume else 'w'
    tee = TeeLogger(log_path, mode=log_mode)
    sys.stdout = tee

    if args.resume:
        print(f"\n--- Resumed from {args.resume} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    print(f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("V6 Phase-First Language Model")
    print("=" * 60)
    print(f"Size: {args.size}")
    print(f"Complex dim: {config.dim} (= {config.dim * 2} real values/position)")
    print(f"SSM state dim: {config.state_dim} (multi-timescale: fast/medium/slow)")
    print(f"Layers: {config.num_layers}")
    print(f"Banks: {config.num_banks} (semantic + context)")
    print(f"Working memory slots: {config.num_wm_slots} (top-k={config.wm_read_topk}, decay={config.wm_slot_decay})")
    print(f"Internal memory slots: {config.num_im_slots} (top-k={config.im_read_topk})")
    if config.use_attention:
        attn_desc = f"every {config.attn_every} layers" if config.attn_every > 0 else "last layer only"
        print(f"PhaseAttention: ENABLED ({attn_desc}, heads={config.attn_num_heads}, window={config.attn_window_size})")
    else:
        print(f"PhaseAttention: DISABLED (attention-free)")
    print(f"Diversity loss: weight={config.diversity_loss_weight}, floor={config.diversity_loss_floor}, margin={config.diversity_margin}")
    print(f"Epochs: {config.max_epochs}")
    gen_info = f"every {args.gen_every} batches, prompt=\"{args.gen_prompt}\"" if args.gen_every > 0 else "off"
    print(f"Mid-epoch generation: {gen_info}")
    print(f"Max samples: {args.max_samples}")
    print(f"Log file: {log_path}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 60)

    train_ds, val_ds, tokenizer = load_tinystories(args.max_samples, args.seq_len)
    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = args.seq_len

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    model = create_model(config)
    init_info = model.initializer_info
    print(f"Init strategy: {init_info['init_strategy']} (seed: {init_info['init_seed']})")

    start_epoch = 0
    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    if config.compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    trainer = Trainer(
        model, config, train_loader, val_loader,
        tokenizer=tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        start_epoch=start_epoch,
    )
    trainer.gen_every = args.gen_every
    trainer.gen_prompt = args.gen_prompt

    if args.resume and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.best_val_loss = best_val_loss
        trainer.best_val_ppl = best_val_ppl

    trainer.train()

    print(f"\nWall clock end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee._stdout
    tee.close()


if __name__ == '__main__':
    main()
