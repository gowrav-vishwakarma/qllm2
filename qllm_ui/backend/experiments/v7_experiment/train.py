"""Auto-generated training script from QLLM Architecture Builder."""
import os
import sys
import time
import math
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import GeneratedModel
from config import config


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return chunk[:-1].long(), chunk[1:].long()


def load_data(cfg):
    """Load and tokenize dataset."""
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_name = cfg["dataset"]
    if dataset_name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split="train")
        text_key = "text"
    elif dataset_name == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        text_key = "text"
    elif dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_key = "text"
    else:
        ds = load_dataset(dataset_name, split="train")
        text_key = list(ds.features.keys())[0]

    all_ids = []
    for example in ds:
        text = example[text_key]
        if text and text.strip():
            ids = tokenizer.encode(text)
            all_ids.extend(ids)
        if len(all_ids) > 50_000_000:
            break

    tokens = torch.tensor(all_ids, dtype=torch.long)
    print(f"Loaded {len(tokens):,} tokens from {dataset_name}")
    return tokens, tokenizer


def train():
    cfg = config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    tokens, tokenizer = load_data(cfg)
    dataset = TextDataset(tokens, cfg["seq_len"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    model = GeneratedModel().to(device)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")

    if cfg.get("compile", False):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )

    total_steps = len(loader) * cfg["epochs"] // cfg.get("grad_accumulation", 1)
    warmup_steps = cfg["warmup_steps"]

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    best_loss = float("inf")
    global_step = 0
    log_path = Path(cfg["log_dir"]) / "training.log"

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", enabled=cfg["amp"] and device.type == "cuda"):
                output = model(input_ids)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            loss_val = loss.item()
            epoch_loss += loss_val * input_ids.size(0)
            epoch_tokens += input_ids.size(0) * input_ids.size(1)

            scaler.scale(loss / cfg.get("grad_accumulation", 1)).backward()

            if (batch_idx + 1) % cfg.get("grad_accumulation", 1) == 0:
                if cfg["grad_clip"] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            if batch_idx % 50 == 0:
                ppl = math.exp(min(loss_val, 20))
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                tok_per_sec = epoch_tokens / max(elapsed, 1e-6)
                msg = f"epoch {epoch} step {batch_idx}/{len(loader)} loss {loss_val:.4f} ppl {ppl:.2f} lr {lr:.2e} tok/s {tok_per_sec:.0f}"
                print(msg, flush=True)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")

        avg_loss = epoch_loss / max(len(dataset), 1)
        avg_ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0
        msg = f"=== Epoch {epoch} done: avg_loss={avg_loss:.4f} ppl={avg_ppl:.2f} time={elapsed:.1f}s ==="
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": cfg,
        }

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], "best_model.pt"))
            print(f"  Saved best model (loss={best_loss:.4f})")

        torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch}.pt"))

    print("Training complete!")


if __name__ == "__main__":
    train()
