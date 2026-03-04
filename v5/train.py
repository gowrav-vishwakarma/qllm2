"""
V5 Training Script.

Matches the reviewer's benchmark: 20k TinyStories, same tokenizer (GPT-2),
same optimizer (AdamW), same schedule (cosine), 20 epochs, small scale.

Usage:
    uv run python -m v5.train --size small --epochs 20 --max_samples 20000
    uv run python -m v5.train --size small-matched --epochs 20  # match ~8M params
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from v5.model import AlgebraicLM, create_model, ModelOutput
from v5.config import V5Config, get_config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Pre-tokenized text dataset for language modeling."""

    def __init__(self, tokens: torch.Tensor, seq_len: int = 512):
        """
        Args:
            tokens: [N] flat token tensor
            seq_len: chunk length
        """
        self.seq_len = seq_len
        n_chunks = len(tokens) // (seq_len + 1)
        self.data = tokens[:n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.data[idx]
        return {
            'input_ids': chunk[:-1],
            'labels': chunk[1:],
        }


def load_tinystories(max_samples: Optional[int] = 20000, seq_len: int = 512):
    """Load TinyStories and tokenize with GPT-2 tokenizer."""
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

    # Split 90/10
    split = int(len(all_tokens) * 0.9)
    train_tokens = all_tokens[:split]
    val_tokens = all_tokens[split:]

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)

    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
    return train_ds, val_ds, tokenizer


def _random_dataset(vocab_size, seq_len, num_samples):
    tokens = torch.randint(1, vocab_size, (num_samples * (seq_len + 1),))
    return TextDataset(tokens, seq_len)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: AlgebraicLM,
        config: V5Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

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

        # AMP only on CUDA
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

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                output = self.model(input_ids)

                # Cross-entropy loss
                logits = output.logits.view(-1, output.logits.size(-1))
                ce_loss = F.cross_entropy(logits, labels.view(-1))

                # Diversity loss
                loss = ce_loss
                div_loss_val = 0.0
                if output.diversity_loss is not None:
                    div_loss = output.diversity_loss * self.config.diversity_loss_weight
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

            if batch_idx % 50 == 0:
                ppl = math.exp(min(ce_loss.item(), 20))
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  [{epoch+1}] batch {batch_idx}/{len(self.train_loader)} "
                    f"loss={ce_loss.item():.4f} ppl={ppl:.1f} "
                    f"div={div_loss_val:.4f} lr={lr:.2e}"
                )

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
        return {
            'val_loss': avg_loss,
            'val_ppl': math.exp(min(avg_loss, 20)),
        }

    def train(self):
        print(f"\nTraining on {self.device}")
        params = self.model.count_parameters()
        print(f"Parameters: {params}")
        print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")
        print(f"Epochs: {self.config.max_epochs}, Batches/epoch: {len(self.train_loader)}")
        print()

        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()

            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            line = (
                f"Epoch {epoch+1}/{self.config.max_epochs} | "
                f"Train Loss: {train_metrics['ce_loss']:.4f} "
                f"PPL: {train_metrics['ppl']:.2f} | "
                f"Time: {epoch_time:.1f}s"
            )

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

            print(line)

        print(f"\nBest Val Loss: {self.best_val_loss:.4f}, Best Val PPL: {self.best_val_ppl:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train V5 Algebraic LM')
    parser.add_argument('--size', type=str, default='small',
                        choices=['tiny', 'small', 'small-matched', 'medium'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max_samples', type=int, default=20000)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--num_banks', type=int, default=None)
    parser.add_argument('--no_attention', action='store_true')
    parser.add_argument('--no_banks', action='store_true')
    args = parser.parse_args()

    config = get_config(args.size)
    config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.num_banks is not None:
        config.num_banks = args.num_banks
    if args.no_attention:
        config.attn_every_k = 0
    if args.no_banks:
        config.num_banks = 0

    print("=" * 60)
    print("V5 Algebraic Language Model")
    print("=" * 60)
    print(f"Size: {args.size}")
    print(f"Complex dim: {config.dim} (= {config.dim * 2} real values/position)")
    print(f"SSM state dim: {config.state_dim}")
    print(f"Layers: {config.num_layers}")
    print(f"Banks: {config.num_banks}")
    print(f"Attention every: {config.attn_every_k} layers (0=none)")
    print(f"Epochs: {config.max_epochs}")
    print(f"Max samples: {args.max_samples}")
    print("=" * 60)

    # Load data
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

    # Create model
    model = create_model(config)

    if config.compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Train
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()

    # Generate sample text
    print("\n" + "=" * 60)
    print("Sample generation:")
    print("=" * 60)
    prompt = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], device=trainer.device)

    model_to_gen = model._orig_mod if hasattr(model, '_orig_mod') else model
    generated = model_to_gen.generate(prompt_tensor, max_new_tokens=100, temperature=0.8, top_k=50)
    text = tokenizer.decode(generated[0].tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated: {text}")


if __name__ == '__main__':
    main()
