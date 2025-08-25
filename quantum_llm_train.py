#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum-inspired LLM – compact, VRAM-friendly, and runnable on RTX 4090 (24GB)

What's inside (practical subset of our ideas):
- Byte-level tokenizer (0..255) for robust cross-dataset training.
- Learned positional embeddings + proper causal attention mask.
- Quantum-inspired PhaseRotator (learned per-dim phase; rotates features).
- LoRA on attention projections (q,k,v,out) for efficient scaling.
- Optional global memory tokens (added to K/V only) for long-range context.
- Mixed precision via torch.amp.autocast("cuda") + GradScaler (new API).
- Dataset selector: wikitext2 / openwebtext / tinystories.
- Per-epoch validation perplexity + best checkpoint saving.
- Generation with temperature, top-k, top-p, repetition penalty, min-p.
- Adaptive grad accumulation (manual via --accumulate_steps; prints guidance).

CLI examples:
  Train (small):
    uv run quantum_llm_train.py --mode train --dataset wikitext2 --max_samples 50000 \
      --epochs 10 --batch_size 32 --seq_length 128 --model_dim 384 --num_layers 6 \
      --num_heads 8 --phase_coh 0.1 --global_tokens 4 --lora_rank 8 --accumulate_steps 2

  Generate:
    uv run quantum_llm_train.py --mode generate --checkpoint checkpoints/checkpoint_last.pt \
      --prompt "The future of AI is" --temperature 0.9 --top_k 50 --top_p 0.95 --repetition_penalty 1.1
"""

import os
import math
import argparse
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# ----------------------------
# Utils
# ----------------------------

def device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def causal_mask(seq_len: int, device: str):
    # [1, 1, L, L] with -inf above diagonal
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def bytes_encode(text: str) -> List[int]:
    return list(text.encode("utf-8", errors="ignore"))  # 0..255


def bytes_decode(ids: List[int]) -> str:
    # decode bytes; replace invalids gracefully
    return bytes([int(x) % 256 for x in ids]).decode("utf-8", errors="ignore")


# ----------------------------
# Dataset (byte-level, streaming chunker)
# ----------------------------

class ByteLMChunked(Dataset):
    """
    Concatenate all text, convert to bytes [0..255], make fixed-length chunks (stride==seq_length).
    """
    def __init__(self, dataset_name: str, split: str, seq_length: int,
                 max_samples: Optional[int] = None):
        assert dataset_name in {"wikitext2", "openwebtext", "tinystories"}
        self.seq_length = seq_length
        texts: List[str] = []

        if dataset_name == "wikitext2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [ex["text"] for ex in ds]
        elif dataset_name == "openwebtext":
            # 10GB+ full set; rely on max_samples to cap
            ds = load_dataset("Skylion007/openwebtext", split=split)
            texts = [ex["text"] for ex in ds]
        elif dataset_name == "tinystories":
            ds = load_dataset("roneneldan/TinyStories", split=split)
            # field is "text"
            texts = [ex["text"] for ex in ds]

        # Join and byte-encode
        joined = "\n\n".join(texts)
        tokens = bytes_encode(joined)

        # Max samples -> cap number of tokens roughly (samples * seq_length)
        if max_samples is not None:
            cap = max_samples * seq_length
            tokens = tokens[: max(cap, seq_length)]

        # Build examples
        n = len(tokens)
        self.x = []
        self.y = []
        for i in range(0, n - seq_length - 1, seq_length):
            chunk = tokens[i : i + seq_length]
            target = tokens[i + 1 : i + 1 + seq_length]
            if len(chunk) == seq_length and len(target) == seq_length:
                self.x.append(torch.tensor(chunk, dtype=torch.long))
                self.y.append(torch.tensor(target, dtype=torch.long))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_loaders(dataset: str, seq_length: int, batch_size: int, max_samples: Optional[int]):
    # train/val split policy:
    # - wikitext2 has 'train'/'validation'
    # - openwebtext has only 'train' -> carve out last 1% for val
    # - tinystories has only 'train' -> carve out last 1% for val
    if dataset == "wikitext2":
        train_ds = ByteLMChunked("wikitext2", "train", seq_length, max_samples)
        val_ds = ByteLMChunked("wikitext2", "validation", seq_length, max_samples=None)
    else:
        full_train = ByteLMChunked(dataset, "train", seq_length, max_samples)
        n = len(full_train)
        val_n = max(64, n // 100)  # 1% or at least 64 batches
        train_n = n - val_n
        # quick split by slicing
        train_ds = torch.utils.data.Subset(full_train, list(range(0, train_n)))
        val_ds = torch.utils.data.Subset(full_train, list(range(train_n, n)))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        drop_last=False,
    )
    return train_loader, val_loader


# ----------------------------
# LoRA
# ----------------------------

class LoRALinear(nn.Module):
    """
    LoRA for Linear layers: y = xW^T + x(AB)^T * (alpha/r)
    Only the A,B params are trainable when lora_rank>0 and lora_train_only=True.
    """
    def __init__(self, in_features, out_features, bias=True, lora_rank=0, lora_alpha=1.0, lora_train_only=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.r = int(lora_rank)
        self.alpha = float(lora_alpha) if self.r > 0 else 1.0
        self.scaling = self.alpha / max(1, self.r)
        self.lora_train_only = lora_train_only

        if self.r > 0:
            self.A = nn.Parameter(torch.zeros(out_features, self.r))
            self.B = nn.Parameter(torch.zeros(self.r, in_features))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        out = self.linear(x)
        if self.r > 0:
            # x @ B^T -> (B @ x^T)^T but we keep shapes consistent
            # implement as (x @ B^T) @ A^T = x @ (B^T @ A^T) = x @ (AB)^T
            lora_update = F.linear(x, self.A.t())   # (N, r)
            lora_update = F.linear(lora_update, self.B.t())  # (N, dim)
            out = out + self.scaling * lora_update
        return out

    def train(self, mode=True):
        super().train(mode)
        if self.r > 0 and self.lora_train_only:
            # freeze base linear
            for p in self.linear.parameters():
                p.requires_grad_(False)
            # train only LoRA params
            if self.A is not None: self.A.requires_grad_(True)
            if self.B is not None: self.B.requires_grad_(True)
        return self


# ----------------------------
# Model
# ----------------------------

class PhaseRotator(nn.Module):
    """Learned per-dim phase rotation: r' = r*cos(phi) - r*sin(phi) on features (acts like complex rotation)."""
    def __init__(self, dim):
        super().__init__()
        self.phase = nn.Parameter(torch.zeros(dim))  # init near 0

    def forward(self, x):
        # x: [B, L, D]
        phi = torch.tanh(self.phase) * math.pi
        c, s = torch.cos(phi), torch.sin(phi)
        return x * c - x * s  # simple rotation-like modulation

    def coherence_loss(self):
        # Encourage smooth/low-variance phases (quantum-inspired coherence)
        diff = self.phase[1:] - self.phase[:-1]
        return (diff**2).mean() + (self.phase**2).mean() * 0.1


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, lora_rank=0, lora_alpha=8.0, lora_train_only=False, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # LoRA-enabled projections
        self.q = LoRALinear(dim, dim, bias=True, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_train_only=lora_train_only)
        self.k = LoRALinear(dim, dim, bias=True, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_train_only=lora_train_only)
        self.v = LoRALinear(dim, dim, bias=True, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_train_only=lora_train_only)
        self.o = LoRALinear(dim, dim, bias=True, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_train_only=lora_train_only)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,hd]

    def _merge(self, x):
        B, H, L, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * hd)

    def forward(self, x, attn_bias=None):
        B, L, D = x.shape
        q = self._split(self.q(x))
        k = self._split(self.k(x))
        v = self._split(self.v(x))
        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,L,L]
        if attn_bias is not None:
            scores = scores + attn_bias  # broadcast [1,1,L,L]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,H,L,hd]
        out = self._merge(out)
        out = self.o(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0,
                 lora_rank=0, lora_alpha=8.0, lora_train_only=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, lora_rank, lora_alpha, lora_train_only, dropout)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias=None):
        x = x + self.attn(self.ln1(x), attn_bias)
        x = x + self.mlp(self.ln2(x))
        return x


class QuantumInspiredLLM(nn.Module):
    def __init__(self, vocab_size=256, dim=384, depth=6, num_heads=8,
                 seq_length=128, global_tokens=0, lora_rank=0, lora_alpha=8.0, lora_train_only=False, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_length = seq_length
        self.global_tokens = int(global_tokens)

        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(seq_length + self.global_tokens, dim)

        self.phase = PhaseRotator(dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio=4.0, dropout=dropout,
                             lora_rank=lora_rank, lora_alpha=lora_alpha, lora_train_only=lora_train_only)
            for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # learned global memory vectors (only added to K/V by prepending to sequence)
        if self.global_tokens > 0:
            self.global_memory = nn.Parameter(torch.randn(self.global_tokens, dim) * 0.02)
        else:
            self.register_parameter("global_memory", None)

    def add_global_to_inputs(self, x):
        # x: [B, L, D] token+pos already applied
        if self.global_tokens <= 0:
            return x, 0
        B = x.size(0)
        g = self.global_memory.unsqueeze(0).expand(B, -1, -1)  # [B,G,D]
        x_aug = torch.cat([g, x], dim=1)  # [B, G+L, D]
        return x_aug, self.global_tokens

    def forward(self, idx):
        # idx: [B, L]
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.tok_embed(idx) + self.pos_embed(pos)  # [B,L,D]
        x = self.phase(x)  # phase rotation

        # optionally prepend global tokens to K/V context
        x, g = self.add_global_to_inputs(x)           # [B, L+g, D]
        total_len = L + g

        # causal bias for total_len (tokens can attend to earlier tokens + globals)
        attn_bias = causal_mask(total_len, idx.device)

        for blk in self.blocks:
            x = blk(x, attn_bias=attn_bias)

        x = self.ln(x)
        # logits only for the last L positions (exclude globals)
        logits = self.head(x[:, g:, :])  # [B,L,V]
        return logits

    def phase_coherence_loss(self):
        return self.phase.coherence_loss()


# ----------------------------
# Training / Evaluation
# ----------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        ntoks = yb.numel()
        total_loss += loss.item() * ntoks
        total_tokens += ntoks
    ppl = math.exp(total_loss / max(1, total_tokens))
    return ppl


def train(args):
    dev = device_str()
    print("Device:", dev)
    torch.manual_seed(args.seed)

    train_loader, val_loader = build_loaders(
        dataset=args.dataset,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    model = QuantumInspiredLLM(
        vocab_size=256,
        dim=args.model_dim,
        depth=args.num_layers,
        num_heads=args.num_heads,
        seq_length=args.seq_length,
        global_tokens=args.global_tokens,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_train_only=args.lora_train_only,
        dropout=args.dropout,
    ).to(dev)

    # If lora_train_only, freeze non-LoRA params proactively
    if args.lora_rank > 0 and args.lora_train_only:
        for n, p in model.named_parameters():
            if "A" in n or "B" in n:
                p.requires_grad_(True)
            else:
                # keep LoRA base Linear frozen, global mem & phase trainable (small)
                if ("tok_embed" in n or "pos_embed" in n or "global_memory" in n or "phase" in n):
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_ppl = float("inf")
    step = 0

    # Simple throughput hint for adaptive accumulation
    print(f"[Hint] Using accumulate_steps={args.accumulate_steps}. "
          f"If VRAM is comfy, try larger --batch_size or --model_dim; "
          f"if VRAM is tight, increase --accumulate_steps.")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_ph = 0.0, 0.0
        tokens_per_sec = 0.0
        import time as _t
        t0 = _t.time()

        for xb, yb in train_loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                logits = model(xb)
                ce = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                ph = model.phase_coherence_loss() * args.phase_coh
                loss = ce + ph

            scaler.scale(loss / args.accumulate_steps).backward()

            if (step + 1) % args.accumulate_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            running_ph += ph.item()
            step += 1

            if step % args.log_every == 0:
                dt = _t.time() - t0
                tps = (args.batch_size * args.seq_length * args.log_every) / max(1e-6, dt)
                tokens_per_sec = tps
                print(f"Epoch {epoch} Step {step}: Loss {running_loss/args.log_every:.4f} "
                      f"(CE {ce.item():.4f} + PH {running_ph/args.log_every:.4f}) | {tps:.1f} tok/s")
                running_loss, running_ph = 0.0, 0.0
                t0 = _t.time()

            if args.save_every > 0 and step % args.save_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"model_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "checkpoint_last.pt"))
                print(f"Saved checkpoint: {ckpt_path} and checkpoint_last.pt")

        # End of epoch: evaluate
        val_ppl = evaluate(model, val_loader, dev)
        print(f"Epoch {epoch} done. Validation Perplexity: {val_ppl:.3f}")
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "checkpoint_last.pt"))

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_perplexity.pt"))
            print(f"New best perplexity: {best_val_ppl:.3f} (saved best_perplexity.pt)")

    print("Training done. Saved checkpoints/checkpoint_last.pt")


# ----------------------------
# Sampling / Generation
# ----------------------------

@torch.no_grad()
def sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0, recent_ids=None, min_p=0.0):
    # logits: [B, V]
    if repetition_penalty != 1.0 and recent_ids is not None:
        # penalize recently used tokens
        for b in range(logits.size(0)):
            for tid in recent_ids[b]:
                logits[b, tid] /= repetition_penalty

    if temperature <= 0:
        # greedy
        return torch.argmax(logits, dim=-1)

    logits = logits / max(1e-8, temperature)

    # Top-k
    if top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        kth = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    # Nucleus (top-p)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cums = torch.cumsum(probs, dim=-1)
        mask = cums > top_p
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.empty_like(logits).scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    # Min-p (truncate tail probabilities)
    if min_p > 0.0:
        probs = torch.softmax(logits, dim=-1)
        logits = torch.where(probs < min_p, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def generate(args):
    dev = device_str()
    model = QuantumInspiredLLM(
        vocab_size=256,
        dim=args.model_dim,
        depth=args.num_layers,
        num_heads=args.num_heads,
        seq_length=args.seq_length,
        global_tokens=args.global_tokens,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_train_only=False,  # inference
        dropout=0.0,
    ).to(dev)

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    model.eval()

    context = bytes_encode(args.prompt)[: args.seq_length]
    if len(context) == 0:
        context = [ord(" ")]
    x = torch.tensor(context, dtype=torch.long, device=dev).unsqueeze(0)  # [1, L<=seq]

    out_ids: List[int] = list(context)
    recent_window = 64

    with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
        for _ in range(args.max_new_tokens):
            # crop to model context window
            x_in = x[:, -args.seq_length:]
            logits = model(x_in)  # [1, L, V]
            next_logits = logits[:, -1, :]  # [1, V]
            next_id = sample_next_token(
                next_logits,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                recent_ids=[out_ids[-recent_window:]],
                min_p=args.min_p,
            ).item()
            out_ids.append(next_id)
            x = torch.tensor(out_ids, dtype=torch.long, device=dev).unsqueeze(0)

    text = bytes_decode(out_ids)
    print("\n---\nGenerated:\n", text)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "generate"])

    # Data / model
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "openwebtext", "tinystories"])
    parser.add_argument("--max_samples", type=int, default=None, help="Approx number of training sequences")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--model_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Phase / LoRA / global memory
    parser.add_argument("--phase_coh", type=float, default=0.1, help="Weight for phase coherence reg")
    parser.add_argument("--global_tokens", type=int, default=0, help="Prepended global memory tokens")
    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank (0 disables)")
    parser.add_argument("--lora_alpha", type=float, default=8.0, help="LoRA scaling alpha")
    parser.add_argument("--lora_train_only", action="store_true", help="Train only LoRA adapters")

    # Optim / training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)

    # Generation
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Hello world")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--min_p", type=float, default=0.0)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
