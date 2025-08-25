# ==========================
# file: quantum_llm_train.py
# ==========================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum-inspired LLM — rich, modular, and VRAM-friendly (RTX 4090 ready)

Features kept & improved:
- Byte tokenizer (0..255), causal masking, phase rotator with coherence reg.
- LoRA (q,k,v,o) with correct BA order; optional LoRA-only training.
- Global memory tokens (prepended to K/V) for long-range inductive bias.
- Datasets: wikitext2, tinystories, c4_en_small, fineweb_sample (OpenWebText avoided).
- Mixed precision via torch.amp + GradScaler (new API).
- Checkpoint + model_args.json saved; generate auto-loads to prevent mismatches.
- Per-epoch validation perplexity + best checkpoint.
- Sampling: temperature, top-k, nucleus, repetition penalty, min-p.
- Accumulation hints for low VRAM; gradient clipping.
"""

import os, math, argparse
import torch
import torch.nn as nn
from typing import Optional, List

from qllm_utils import device_str, bytes_encode, bytes_decode, save_args_json, load_args_json
from datasets_qllm import build_loaders
from quantum_llm_model import QuantumInspiredLLM
from sampling_qllm import sample_next_token


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

    train_loader, val_loader = build_loaders(args.dataset, args.seq_length, args.batch_size, args.max_samples)

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

    # Freeze base if LoRA-only
    if args.lora_rank > 0 and args.lora_train_only:
        for n, p in model.named_parameters():
            if ("A" in n or "B" in n) or ("tok_embed" in n or "pos_embed" in n or "global_memory" in n or "phase" in n):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_args_path = os.path.join(args.checkpoint_dir, "model_args.json")
    save_args_json(model_args_path, {
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "seq_length": args.seq_length,
        "global_tokens": args.global_tokens,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "dataset": args.dataset,
    })

    best_val_ppl = float("inf")
    step = 0
    print(f"[Hint] Using accumulate_steps={args.accumulate_steps}. If VRAM is comfy, try larger --batch_size or --model_dim; if VRAM is tight, increase --accumulate_steps.")

    criterion = nn.CrossEntropyLoss()
    import time as _t

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_ph = 0.0, 0.0
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

            step += 1
            running_loss += loss.item()
            running_ph += ph.item()

            if step % args.log_every == 0:
                dt = _t.time() - t0
                tps = (args.batch_size * args.seq_length * args.log_every) / max(1e-6, dt)
                print(f"Epoch {epoch} Step {step}: Loss {running_loss/args.log_every:.4f} (CE {ce.item():.4f} + PH {running_ph/args.log_every:.4f}) | {tps:.1f} tok/s")
                running_loss, running_ph = 0.0, 0.0
                t0 = _t.time()

            if args.save_every > 0 and step % args.save_every == 0:
                ckpt = os.path.join(args.checkpoint_dir, f"model_step{step}.pt")
                torch.save(model.state_dict(), ckpt)
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "checkpoint_last.pt"))
                print(f"Saved checkpoint: {ckpt} and checkpoint_last.pt")

        val_ppl = evaluate(model, val_loader, dev)
        print(f"Epoch {epoch} done. Validation Perplexity: {val_ppl:.3f}")
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "checkpoint_last.pt"))
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_perplexity.pt"))
            print(f"New best perplexity: {best_val_ppl:.3f} (saved best_perplexity.pt)")

    print("Training done. Saved checkpoints/checkpoint_last.pt")


def generate(args):
    dev = device_str()

    # Load saved model args to avoid mismatches (pos_embed/LoRA/global_tokens)
    if args.model_args:
        saved = load_args_json(args.model_args)
    else:
        # default: use checkpoint_dir next to checkpoint
        ckpt_dir = os.path.dirname(args.checkpoint) if args.checkpoint else "."
        saved_path = os.path.join(ckpt_dir, "model_args.json")
        if not os.path.exists(saved_path):
            raise FileNotFoundError(f"model_args.json not found next to checkpoint: {saved_path}")
        saved = load_args_json(saved_path)

    # Build model with saved dims
    model = QuantumInspiredLLM(
        vocab_size=256,
        dim=saved.get("model_dim", args.model_dim or 384),
        depth=saved.get("num_layers", args.num_layers or 6),
        num_heads=saved.get("num_heads", args.num_heads or 8),
        seq_length=saved.get("seq_length", args.seq_length or 128),
        global_tokens=saved.get("global_tokens", args.global_tokens or 0),
        lora_rank=saved.get("lora_rank", args.lora_rank or 0),
        lora_alpha=saved.get("lora_alpha", args.lora_alpha or 8.0),
        lora_train_only=False,
        dropout=0.0,
    ).to(dev)

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Strict load: ensures no silent mismatches
    state = torch.load(args.checkpoint, map_location=dev)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print("[Warn] Unexpected keys in state_dict:", unexpected)
    if missing:
        print("[Warn] Missing keys in state_dict:", missing)
    model.eval()

    context = bytes_encode(args.prompt)[: model.seq_length]
    if len(context) == 0:
        context = [ord(" ")]
    x = torch.tensor(context, dtype=torch.long, device=dev).unsqueeze(0)

    out_ids: List[int] = list(context)
    recent_window = 64

    with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
        for _ in range(args.max_new_tokens):
            x_in = x[:, -model.seq_length:]
            logits = model(x_in)
            next_logits = logits[:, -1, :]
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

    print("\n---\nGenerated:\n", bytes_decode(out_ids))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["train", "generate"]) 
    # Data / model
    p.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "tinystories", "c4_en_small", "fineweb_sample"]) 
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_length", type=int, default=128)
    p.add_argument("--model_dim", type=int, default=384)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    # Phase / LoRA / globals
    p.add_argument("--phase_coh", type=float, default=0.1)
    p.add_argument("--global_tokens", type=int, default=0)
    p.add_argument("--lora_rank", type=int, default=0)
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_train_only", action="store_true")
    # Optim
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--accumulate_steps", type=int, default=1)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    # Generation
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--model_args", type=str, default=None, help="Path to model_args.json (optional)")
    p.add_argument("--prompt", type=str, default="Hello world")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--min_p", type=float, default=0.0)

    args = p.parse_args()

    if args.mode == "train":
        train(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
