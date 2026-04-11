"""
RPAM PyTorch training — WikiText-103.
Usage: python -m v6.train_real --size <config>
"""

import sys, os, time, math, argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .model_real import RPAMModel
from .config import get_config


def load_wikitext103(seq_len, split="train", max_samples=None):
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    texts = ds["text"]
    if max_samples:
        texts = texts[:max_samples]
    full_text = "\n".join(t for t in texts if t.strip())
    tokens = tokenizer.encode(full_text)
    print(f"  {split}: {len(tokens):,} tokens")
    n_chunks = len(tokens) // seq_len
    tokens = tokens[:n_chunks * seq_len]
    chunks = np.array(tokens, dtype=np.int64).reshape(n_chunks, seq_len)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="rpam-100m",
                        choices=["rpam-5m", "rpam-10m", "rpam-50m", "rpam-100m", "rpam-1b"])
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--expand", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(args.size)
    dim = args.dim or cfg.dim
    layers = args.layers or cfg.num_layers
    heads = args.heads or cfg.pam_num_heads
    head_dim = args.head_dim or cfg.pam_head_dim
    expand = args.expand or cfg.bank_expand
    batch_size = args.batch_size or cfg.batch_size
    lr = args.lr or cfg.learning_rate
    save_dir = args.save_dir or f"checkpoints_{args.size}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RPAM PyTorch Training")
    print(f"  Device: {device}")
    print(f"  Config: dim={dim}, layers={layers}, heads={heads}, head_dim={head_dim}")
    print(f"  Training: seq_len={args.seq_len}, batch_size={batch_size}, epochs={args.epochs}")
    print(f"  LR: {lr}, warmup: {args.warmup}")

    print("Loading WikiText-103...")
    train_data = load_wikitext103(args.seq_len + 1, "train")
    val_data = load_wikitext103(args.seq_len + 1, "validation")
    print(f"  Train chunks: {len(train_data)}, Val chunks: {len(val_data)}")

    train_ds = TensorDataset(torch.from_numpy(train_data))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = RPAMModel(
        vocab_size=50257, dim=dim, num_layers=layers,
        expand=expand, num_heads=heads, head_dim=head_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    n_batches = len(train_loader)
    total_steps = n_batches * args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    from torch.optim.lr_scheduler import LambdaLR
    import math as m

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1.0 + m.cos(m.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    os.makedirs(save_dir, exist_ok=True)
    best_val_ppl = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, (batch,) in enumerate(train_loader):
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 50257), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * x.numel()
            epoch_tokens += x.numel()

            if batch_idx % args.log_every == 0:
                ppl = math.exp(min(loss.item(), 20))
                tok_s = epoch_tokens / max(1, time.time() - t0)
                lr_now = scheduler.get_last_lr()[0]
                print(f"  [{epoch}] {batch_idx}/{n_batches} loss={loss.item():.4f} ppl={ppl:.1f} "
                      f"lr={lr_now:.2e} | {tok_s:.0f} tok/s")

            if batch_idx > 0 and batch_idx % args.val_every == 0:
                val_ppl = evaluate(model, val_data, batch_size, args.seq_len, device)
                print(f"  ** Val ppl={val_ppl:.2f}")
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                    print(f"  ** Saved best model (val PPL {val_ppl:.2f})")

        elapsed = time.time() - t0
        avg_loss = epoch_loss / epoch_tokens
        train_ppl = math.exp(min(avg_loss, 20))
        val_ppl = evaluate(model, val_data, batch_size, args.seq_len, device)
        print(f"\n  Epoch {epoch}: train PPL={train_ppl:.2f}, val PPL={val_ppl:.2f}, "
              f"time={elapsed:.0f}s ({epoch_tokens/elapsed:.0f} tok/s)")
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pt"))
        old = os.path.join(save_dir, f"epoch_{epoch-2}.pt")
        if os.path.exists(old):
            os.remove(old)

    print(f"\nDone. Best val PPL: {best_val_ppl:.2f}")


def evaluate(model, val_data, batch_size, seq_len, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = min(len(val_data) // batch_size, max_batches)
    with torch.no_grad():
        for i in range(n_batches):
            batch = torch.from_numpy(val_data[i*batch_size:(i+1)*batch_size])
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 50257), y.view(-1))
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    model.train()
    return math.exp(min(total_loss / total_tokens, 20))


if __name__ == "__main__":
    main()