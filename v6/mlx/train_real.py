"""
RPAM MLX training script — WikiText-103 (real-valued ablation).

Usage:
    cd /Users/caug/npcww/qnlp/qllm-private
    uv run python /Users/caug/npcww/qnlp/ket-nlp/qpam_mlx/train_real.py
"""

import sys, os, time, math, argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Add parent for model import
sys.path.insert(0, os.path.dirname(__file__))
from model_real import RPAMModel


def load_wikitext103(seq_len: int, split: str = "train", max_samples: int = None):
    """Load and tokenize WikiText-103 using HF tokenizer."""
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    # Concatenate all text
    texts = ds["text"]
    if max_samples:
        texts = texts[:max_samples]
    full_text = "\n".join(t for t in texts if t.strip())
    tokens = tokenizer.encode(full_text)
    print(f"  {split}: {len(tokens):,} tokens")

    # Chunk into sequences
    n_chunks = len(tokens) // seq_len
    tokens = tokens[:n_chunks * seq_len]
    chunks = np.array(tokens, dtype=np.int32).reshape(n_chunks, seq_len)
    return chunks


def get_batch(data, batch_idx, batch_size):
    """Get a batch of input/target pairs."""
    start = batch_idx * batch_size
    end = min(start + batch_size, len(data))
    batch = mx.array(data[start:end])
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y


def loss_fn(model, x, y):
    logits = model(x)  # [B, T, V]
    # Cross entropy
    logits_flat = mx.reshape(logits, (-1, logits.shape[-1]))
    targets_flat = mx.reshape(y, (-1,))
    loss = mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=576)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--heads", type=int, default=9)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--expand", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints_rpam_mlx")
    args = parser.parse_args()

    print(f"RPAM MLX Training (Real-valued ablation)")
    print(f"  Device: {mx.default_device()}")
    print(f"  Config: dim={args.dim}, layers={args.layers}, heads={args.heads}, head_dim={args.head_dim}")
    print(f"  Training: seq_len={args.seq_len}, batch_size={args.batch_size}, epochs={args.epochs}")
    print(f"  LR: {args.lr}, warmup: {args.warmup}")
    print()

    # Load data
    print("Loading WikiText-103...")
    train_data = load_wikitext103(args.seq_len + 1, "train", args.max_train)
    val_data = load_wikitext103(args.seq_len + 1, "validation", args.max_val)
    print(f"  Train chunks: {len(train_data)}, Val chunks: {len(val_data)}")
    print()

    # Model
    model = RPAMModel(
        vocab_size=50257, dim=args.dim, num_layers=args.layers,
        expand=args.expand, num_heads=args.heads, head_dim=args.head_dim,
    )

    # Count params
    params = model.parameters()
    def count_params(tree):
        if isinstance(tree, mx.array):
            return tree.size
        elif isinstance(tree, dict):
            return sum(count_params(v) for v in tree.values())
        elif isinstance(tree, list):
            return sum(count_params(v) for v in tree)
        return 0
    n_params = count_params(params)
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print()

    # Optimizer with warmup + cosine decay
    n_batches_per_epoch = len(train_data) // args.batch_size
    total_steps = n_batches_per_epoch * args.epochs

    schedule = optim.cosine_decay(args.lr, total_steps - args.warmup)
    warmup_schedule = optim.linear_schedule(1e-7, args.lr, args.warmup)
    lr_schedule = optim.join_schedules(
        [warmup_schedule, schedule], [args.warmup]
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    os.makedirs(args.save_dir, exist_ok=True)
    val_log_path = os.path.join(args.save_dir, "val_ppl.log")
    def log_val(epoch, step, val_ppl, kind):
        with open(val_log_path, "a") as f:
            f.write(f"{epoch}\t{step}\t{kind}\t{val_ppl:.4f}\n")
    best_val_ppl = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"{'='*60}")
        print(f"  Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Shuffle
        perm = np.random.permutation(len(train_data))
        train_data_shuffled = train_data[perm]

        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()

        for batch_idx in range(n_batches_per_epoch):
            global_step += 1
            x, y = get_batch(train_data_shuffled, batch_idx, args.batch_size)

            loss, grads = loss_and_grad(model, x, y)
            # Gradient clipping
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_val = loss.item()
            batch_tokens = x.shape[0] * x.shape[1]
            epoch_loss += loss_val * batch_tokens
            epoch_tokens += batch_tokens

            if batch_idx % args.log_every == 0:
                elapsed = time.time() - epoch_start
                tok_per_sec = epoch_tokens / elapsed if elapsed > 0 else 0
                ppl = math.exp(min(loss_val, 20))
                lr_now = optimizer.learning_rate.item() if hasattr(optimizer.learning_rate, 'item') else args.lr
                print(f"  [{epoch}] {batch_idx}/{n_batches_per_epoch} "
                      f"loss={loss_val:.4f} ppl={ppl:.1f} "
                      f"lr={lr_now:.2e} | {tok_per_sec:.0f} tok/s")

            # Validation
            if batch_idx > 0 and batch_idx % args.val_every == 0:
                val_loss = evaluate(model, val_data, args.batch_size, args.seq_len)
                val_ppl = math.exp(min(val_loss, 20))
                print(f"  ** Val loss={val_loss:.4f} ppl={val_ppl:.2f}")
                log_val(epoch, global_step, val_ppl, "mid")
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    # Save
                    save_path = os.path.join(args.save_dir, "best_model.npz")
                    model.save_weights(save_path)
                    print(f"  ** Saved best model (val PPL {val_ppl:.2f})")

        # End of epoch validation
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / epoch_tokens
        train_ppl = math.exp(min(avg_loss, 20))

        val_loss = evaluate(model, val_data, args.batch_size, args.seq_len)
        val_ppl = math.exp(min(val_loss, 20))
        log_val(epoch, global_step, val_ppl, "end")

        print(f"\n  Epoch {epoch} complete: train PPL={train_ppl:.2f}, val PPL={val_ppl:.2f}, "
              f"time={epoch_time:.0f}s ({epoch_tokens/epoch_time:.0f} tok/s)")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            save_path = os.path.join(args.save_dir, "best_model.npz")
            model.save_weights(save_path)
            print(f"  ** New best val PPL: {val_ppl:.2f}")

        epoch_path = os.path.join(args.save_dir, f"epoch_{epoch}.npz")
        model.save_weights(epoch_path)
        print(f"  ** Saved epoch {epoch} checkpoint")

    print(f"\nTraining complete. Best val PPL: {best_val_ppl:.2f}")


def evaluate(model, val_data, batch_size, seq_len):
    """Evaluate on validation set."""
    total_loss = 0.0
    total_tokens = 0
    n_batches = len(val_data) // batch_size

    for i in range(min(n_batches, 50)):  # cap at 50 batches for speed
        x, y = get_batch(val_data, i, batch_size)
        logits = model(x)
        logits_flat = mx.reshape(logits, (-1, logits.shape[-1]))
        targets_flat = mx.reshape(y, (-1,))
        loss = mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))
        mx.eval(loss)
        batch_tokens = x.shape[0] * x.shape[1]
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    return total_loss / total_tokens if total_tokens > 0 else float("inf")


if __name__ == "__main__":
    main()
