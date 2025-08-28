#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_improved.py
Convenience wrapper to launch improved training with recommended defaults.

Usage:
  # single GPU - LARGE SCALE TRAINING
  python train_improved.py

  # generate only (after training)
  python train_improved.py --mode generate --checkpoint checkpoints_improved/best_perplexity.pt

  # torchrun (multi-GPU)
  torchrun --nproc_per_node=2 train_improved.py
"""
import argparse
import os
import sys
from types import SimpleNamespace

# Import the train / generate functions from the main trainer
from quantum_llm_train import train, generate

def build_args_from_defaults(cli_args):
    # LARGE SCALE DEFAULTS for 24GB VRAM utilization (FINE-TUNED FOR MEMORY)
    defaults = {
        "mode": "train",
        "dataset": "wikitext2",    # Fallback to wikitext2 for now
        "max_samples": 500000,     # More samples for better training
        "batch_size": 6,           # Reduced from 8 to fit in memory
        "seq_length": 1024,        # 2x longer sequences
        "model_dim": 2048,         # 2x larger model dimension
        "num_layers": 32,          # 2x more layers
        "num_heads": 32,           # 2x more attention heads
        "dropout": 0.1,
        "phase_coh": 0.05,
        "global_tokens": 32,       # 2x more global tokens
        "lora_rank": 64,           # 2x higher LoRA rank
        "lora_alpha": 128.0,       # 2x higher alpha
        "lora_train_only": False,
        "lr": 2e-5,                # Lower LR for larger model stability
        "weight_decay": 0.01,
        "accumulate_steps": 10,    # Effective batch size = 6 * 10 = 60
        "checkpoint_dir": "checkpoints_large_scale",
        "save_every": 500,         # Save more frequently
        "log_every": 50,           # Log more frequently
        "seed": 42,
        "epochs": 10,              # Fewer epochs due to more data
        "grad_clip": 1.0,
        "activation_checkpoint": True,  # CRITICAL for memory efficiency
        "streaming": False,        # Disable streaming for now
        "num_workers": 8,          # More workers for larger batches
        "warmup_steps": 3000,      # Longer warmup for larger model
        "attention_type": "interference",  # Use quantum interference attention
    }

    # override defaults with CLI-provided values — but only when CLI actually provided a value
    args = defaults.copy()
    for k, v in vars(cli_args).items():
        # argparse store_true defaults to None now (we set defaults in parser)
        if v is not None:
            args[k] = v
    
    # Special handling for boolean flags
    if cli_args.activation_checkpoint is not None:
        args["activation_checkpoint"] = cli_args.activation_checkpoint
    if cli_args.lora_train_only is not None:
        args["lora_train_only"] = cli_args.lora_train_only

    # Attaching derived placeholders expected by quantum_llm_train (if not present)
    args["max_steps_per_epoch"] = args.get("max_steps_per_epoch", 1000)

    # convert to SimpleNamespace
    return SimpleNamespace(**args)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "generate"], default=None)
    p.add_argument("--checkpoint", default=None, help="Path to checkpoint when generating")
    p.add_argument("--attention_type", choices=["classical","interference"], default=None)
    # Make flags default to None so we can detect whether user provided them
    p.add_argument("--lora_train_only", action="store_true", default=None)
    p.add_argument("--activation_checkpoint", action="store_true", default=None)
    # allow overriding a few hyperparams quickly
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--model_dim", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--seq_length", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None)

    args = p.parse_args()
    final_args = build_args_from_defaults(args)

    # ensure checkpoint dir exists
    os.makedirs(final_args.checkpoint_dir, exist_ok=True)

    if final_args.mode == "generate":
        if not final_args.checkpoint:
            print("Provide --checkpoint path for generation. Exiting.")
            sys.exit(1)
        # call generate() from quantum_llm_train
        generate(final_args)
        return

    # train mode (default)
    print("🚀 LAUNCHING LARGE-SCALE QUANTUM LLM TRAINING")
    print("=" * 60)
    print(f"🎯 Model: {final_args.model_dim}d, {final_args.num_layers} layers, {final_args.num_heads} heads")
    print(f"📏 Sequence: {final_args.seq_length} tokens, Global tokens: {final_args.global_tokens}")
    print(f"🧠 Attention: {final_args.attention_type} (quantum interference)")
    print(f"📊 Batch: {final_args.batch_size} × {final_args.accumulate_steps} = {final_args.batch_size * final_args.accumulate_steps} effective")
    print(f"💾 Memory: Activation checkpointing: {final_args.activation_checkpoint}")
    print(f"📚 Dataset: {final_args.dataset} ({final_args.max_samples:,} samples)")
    print(f"🔄 Streaming: {final_args.streaming}")
    print(f"💾 Checkpoint dir: {final_args.checkpoint_dir}")
    print()
    
    # Estimate VRAM usage
    estimated_vram_gb = estimate_vram_usage(final_args)
    print(f"💻 Estimated VRAM usage: ~{estimated_vram_gb:.1f}GB / 24GB ({estimated_vram_gb/24*100:.1f}%)")
    print()

    train(final_args)

def estimate_vram_usage(args):
    """Realistic estimate of VRAM usage for the model (rough)"""
    vocab_size = 256
    # parameter bytes (very rough)
    model_param_bytes = (
        vocab_size * args.model_dim * 4 +
        (args.seq_length + args.global_tokens) * args.model_dim * 4 +
        args.model_dim * args.model_dim * 4 * args.num_layers * 0.5 +  # scaled down estimate
        args.global_tokens * args.model_dim * 4
    )
    activation_per_sample = (
        args.model_dim * args.seq_length * 4 +
        args.model_dim * args.seq_length * 4 * (args.num_heads / max(1, args.num_heads)) +
        args.model_dim * args.seq_length * 4 * args.num_layers * 0.5
    )
    batch_memory = activation_per_sample * args.batch_size
    checkpoint_factor = 0.4 if args.activation_checkpoint else 1.0
    total_bytes = model_param_bytes + (batch_memory * checkpoint_factor)
    return total_bytes / (1024**3)

if __name__ == "__main__":
    main()
