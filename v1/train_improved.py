#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_improved.py - safer defaults for consumer GPU (24GB).
Defaults: LoRA-only training enabled, moderate model size, activation checkpointing on.
"""
import argparse
import os
import sys
from types import SimpleNamespace

from quantum_llm_train import train, generate

def build_args_from_defaults(cli_args):
    defaults = {
        "mode": "train",
        "dataset": "wikitext2",
        "max_samples": 200000,
        "batch_size": 4,
        "seq_length": 256,
        "model_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "dropout": 0.1,
        "phase_coh": 0.02,
        "global_tokens": 8,
        "lora_rank": 32,
        "lora_alpha": 64.0,
        "lora_train_only": False,      # DEFAULT: LoRA-only to save optimizer memory
        "lr": 3e-4,                   # somewhat higher to speed learning with LoRA
        "weight_decay": 0.01,
        "accumulate_steps": 4,        # effective batch = 4 * 4 = 16
        "checkpoint_dir": "checkpoints_improved",
        "save_every": 1000,
        "log_every": 100,
        "seed": 42,
        "epochs": 20,
        "grad_clip": 1.0,
        "activation_checkpoint": True,
        "streaming": False,
        "num_workers": 4,
        "warmup_steps": 1000,
        "attention_type": "interference",
        # interference knobs
        "interference_beta": 0.05,
        "inter_heads_frac": 0.25,
        "compile": False,
    }
    args = defaults.copy()
    for k, v in vars(cli_args).items():
        if v is not None:
            args[k] = v
    args["max_steps_per_epoch"] = args.get("max_steps_per_epoch", 1000)
    return SimpleNamespace(**args)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "generate"], default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--attention_type", choices=["classical","interference"], default=None)
    # set to None so wrapper can detect if user passed flag
    p.add_argument("--lora_train_only", action="store_true", default=None)
    p.add_argument("--activation_checkpoint", action="store_true", default=None)
    p.add_argument("--compile", action="store_true", default=None)

    # quick overrides
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--model_dim", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--seq_length", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None)
    # interference knobs
    p.add_argument("--interference_beta", type=float, default=None)
    p.add_argument("--inter_heads_frac", type=float, default=None)
    # lr/warmup
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--accumulate_steps", type=int, default=None)

    args = p.parse_args()
    final_args = build_args_from_defaults(args)

    os.makedirs(final_args.checkpoint_dir, exist_ok=True)

    if final_args.mode == "generate":
        if not final_args.checkpoint:
            print("Provide --checkpoint path for generation. Exiting.")
            sys.exit(1)
        generate(final_args)
        return

    print("🚀 LAUNCHING IMPROVED QUANTUM-INSPIRED TRAINING")
    print("=" * 60)
    print(f"🎯 Model: {final_args.model_dim}d, {final_args.num_layers} layers, {final_args.num_heads} heads")
    print(f"📏 Sequence: {final_args.seq_length} tokens, Global tokens: {final_args.global_tokens}")
    print(f"🧠 Attention: {final_args.attention_type} (interference_beta={final_args.interference_beta}, heads_frac={final_args.inter_heads_frac})")
    print(f"📊 Batch: {final_args.batch_size} × {final_args.accumulate_steps} = {final_args.batch_size * final_args.accumulate_steps} effective")
    print(f"💾 Memory: Activation checkpointing: {final_args.activation_checkpoint} | compile: {final_args.compile}")
    print(f"📚 Dataset: {final_args.dataset} ({final_args.max_samples:,} samples)")
    print(f"🔄 Streaming: {final_args.streaming}")
    print(f"💾 Checkpoint dir: {final_args.checkpoint_dir}")
    print()

    train(final_args)

if __name__ == "__main__":
    main()
