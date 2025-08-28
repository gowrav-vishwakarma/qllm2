# ==========================
# file: quantum_llm_train.py
# ==========================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
import argparse
import time
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from qllm_utils import (
    device_str, bytes_encode, bytes_decode,
    save_args_json, load_args_json,
    ddp_setup, get_rank, save_checkpoint, get_world_size
)
from datasets_qllm import build_loaders
from quantum_llm_model import QuantumInspiredLLM
from sampling_qllm import sample_next_token

# Silence noisy scheduler warning (we call scheduler AFTER optimizer.step in this script)
warnings.filterwarnings(
    "ignore",
    message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`.*",
    category=UserWarning,
    module="torch.optim.lr_scheduler"
)


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


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


def _count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model_and_opt(args, device, world_size=1):
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
        attention_type=args.attention_type,
        use_checkpoint=args.activation_checkpoint,
        interference_beta=getattr(args, "interference_beta", 0.08),
        inter_heads_fraction=getattr(args, "inter_heads_frac", 1.0),
    ).to(device)

    # === FREEZE NON-LoRA PARAMS WHEN LoRA-ONLY TRAINING IS REQUESTED ===
    if args.lora_rank > 0 and args.lora_train_only:
        # Keep only LoRA A/B trainable. This drastically reduces optimizer state size.
        # LoRA param names often end in '.A' or '.B', but to be robust also match 'A' or 'B' segments.
        kept = 0
        for n, p in model.named_parameters():
            # allowlist only names containing '.A' or '.B' or ending with 'A'/'B'
            name_last = n.split('.')[-1]
            if ".A" in n or ".B" in n or name_last == "A" or name_last == "B":
                p.requires_grad_(True)
                kept += p.numel()
            else:
                p.requires_grad_(False)
        if get_rank() == 0:
            total, trainable = _count_trainable_params(model)
            print(f"[LoRA-only] total params: {total:,}, trainable (LoRA A/B): {trainable:,} (~{trainable/total*100:.4f}%)")
    else:
        # default: all params trainable
        for p in model.parameters():
            p.requires_grad_(True)
        if get_rank() == 0:
            total, trainable = _count_trainable_params(model)
            print(f"[Full-train] total params: {total:,}, trainable: {trainable:,}")

    # Optimizer only over parameters with requires_grad True (so LoRA-only -> tiny optimizer state)
    optim_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    total_steps = max(1, (args.epochs * max(1, args.max_steps_per_epoch)) // (max(1, get_world_size()) * max(1, args.accumulate_steps)))
    warmup = int(args.warmup_steps)
    scheduler = cosine_warmup_scheduler(optimizer, warmup, total_steps)

    # Optional compile for modern PyTorch
    if getattr(args, "compile", False):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception as e:
            if get_rank() == 0:
                print(f"[Warn] torch.compile unavailable or failed: {e}")

    return model, optimizer, scheduler


def train(args):
    ddp_inited = ddp_setup()
    rank = get_rank()
    world_size = get_world_size()

    dev = device_str()
    device_type = "cuda" if dev == "cuda" else ("mps" if dev == "mps" else "cpu")

    # enable TF32 matmul precision for improved speed on supported GPUs (optional)
    try:
        # recommended modern API
        torch.set_float32_matmul_precision('high')
    except Exception:
        # older torch versions may not have this API — ignore if unavailable
        pass

    print(f"Device: {dev} | Rank: {rank} | World size: {world_size}")
    torch.manual_seed(args.seed + rank)

    train_loader, val_loader = build_loaders(args.dataset, args.seq_length, args.batch_size, args.max_samples,
                                            streaming=args.streaming, num_workers=args.num_workers, drop_last=True)

    steps_per_epoch = len(train_loader)
    args.max_steps_per_epoch = steps_per_epoch

    model, optimizer, scheduler = build_model_and_opt(args, dev, world_size)

    if ddp_inited:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device_type == "cuda" else None,
            find_unused_parameters=False
        )

    scaler = torch.amp.GradScaler(enabled=(device_type == "cuda"))

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
        "attention_type": args.attention_type,
        "interference_beta": getattr(args, "interference_beta", 0.08),
        "inter_heads_frac": getattr(args, "inter_heads_frac", 1.0),
    })

    best_val_ppl = float("inf")
    step = 0
    global_step = 0
    print(f"[Hint] accumulate_steps={args.accumulate_steps}. Activation checkpointing: {args.activation_checkpoint}")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_ph = 0.0, 0.0
        t0 = time.time()

        for xb, yb in train_loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                logits = model(xb)
                ce = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                ph_module = model.module if hasattr(model, "module") else model
                ph = ph_module.phase_coherence_loss() * args.phase_coh
                loss = ce + ph

            scaler.scale(loss / args.accumulate_steps).backward()

            if (step + 1) % args.accumulate_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

                # step optimizer & scaler
                try:
                    scaler.step(optimizer)
                except Exception as ex:
                    print(f"[Error] optimizer.step failed: {ex}")
                    raise
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # scheduler AFTER optimizer.step
                try:
                    scheduler.step()
                except Exception as e:
                    if rank == 0:
                        print(f"[Warn] scheduler.step() raised: {e}")

                global_step += 1

            step += 1
            running_loss += float(loss.item())
            running_ph += float(ph.item()) if isinstance(ph, torch.Tensor) else float(ph)
            
            if step % args.log_every == 0 and rank == 0:
                dt = time.time() - t0
                tps = (args.batch_size * args.seq_length * args.log_every) / max(1e-6, dt)

                # try to fetch phase param (works whether model is DDP-wrapped or not)
                core = model.module if hasattr(model, "module") else model
                phase_param = None
                try:
                    phase_param = core.phase.phase
                except Exception:
                    # fallback: search named params
                    for n, p in core.named_parameters():
                        if 'phase' in n:
                            phase_param = p
                            break

                if phase_param is not None:
                    ph_mean = phase_param.data.mean().item()
                    ph_std = phase_param.data.std().item()
                    ph_grad_norm = phase_param.grad.norm().item() if phase_param.grad is not None else 0.0
                    print(f"Epoch {epoch} Step {step}: Loss {running_loss/args.log_every:.4f} (CE {ce.item():.4f} + PH {running_ph/args.log_every:.6f}) | {tps:.1f} tok/s | phase mean {ph_mean:.6f} std {ph_std:.6f} grad_norm {ph_grad_norm:.6f}")
                else:
                    print(f"Epoch {epoch} Step {step}: Loss {running_loss/args.log_every:.4f} (CE {ce.item():.4f} + PH {running_ph/args.log_every:.6f}) | {tps:.1f} tok/s")

                running_loss, running_ph = 0.0, 0.0
                t0 = time.time()


            if args.save_every > 0 and step % args.save_every == 0 and rank == 0:
                ckpt = os.path.join(args.checkpoint_dir, f"model_step{step}.pt")
                state = model.module.state_dict() if ddp_inited else model.state_dict()
                save_checkpoint(state, ckpt, rank=rank)
                save_checkpoint(state, os.path.join(args.checkpoint_dir, "checkpoint_last.pt"), rank=rank)
                print(f"Saved checkpoint: {ckpt} and checkpoint_last.pt")

        if rank == 0:
            core_model = model.module if ddp_inited else model
            val_ppl = evaluate(core_model, val_loader, dev)
            print(f"Epoch {epoch} done. Validation Perplexity: {val_ppl:.3f}")
            save_checkpoint(core_model.state_dict(), os.path.join(args.checkpoint_dir, "checkpoint_last.pt"), rank=rank)
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                save_checkpoint(core_model.state_dict(), os.path.join(args.checkpoint_dir, "best_perplexity.pt"), rank=rank)
                print(f"New best perplexity: {best_val_ppl:.3f} (saved best_perplexity.pt)")

    if rank == 0:
        print("Training done. Saved checkpoints/checkpoint_last.pt")


def generate(args):
    dev = device_str()
    device_type = "cuda" if dev == "cuda" else ("mps" if dev == "mps" else "cpu")

    if args.model_args:
        saved = load_args_json(args.model_args)
    else:
        ckpt_dir = os.path.dirname(args.checkpoint) if args.checkpoint else "."
        saved_path = os.path.join(ckpt_dir, "model_args.json")
        if not os.path.exists(saved_path):
            raise FileNotFoundError(f"model_args.json not found next to checkpoint: {saved_path}")
        saved = load_args_json(saved_path)

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
        attention_type=saved.get("attention_type", args.attention_type or "classical"),
        use_checkpoint=False,
        interference_beta=saved.get("interference_beta", 0.08),
        inter_heads_fraction=saved.get("inter_heads_frac", 1.0),
    ).to(dev)

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

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

    with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
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
    # Quantum/LoRA
    p.add_argument("--phase_coh", type=float, default=0.1)
    p.add_argument("--global_tokens", type=int, default=0)
    p.add_argument("--lora_rank", type=int, default=0)
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_train_only", action="store_true")
    p.add_argument("--attention_type", type=str, default="classical", choices=["classical", "interference"])
    p.add_argument("--interference_beta", type=float, default=0.08)
    p.add_argument("--inter_heads_frac", type=float, default=1.0)
    # Optim & training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--accumulate_steps", type=int, default=1)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--activation_checkpoint", action="store_true", help="Use activation checkpointing to save memory")
    p.add_argument("--streaming", action="store_true", help="Use dataset streaming when available")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps_per_epoch", type=int, default=1000)
    p.add_argument("--compile", action="store_true")
    # generation
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--model_args", type=str, default=None)
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
