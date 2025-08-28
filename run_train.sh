#!/usr/bin/env bash
# run_train.sh - examples for single-GPU and multi-GPU launches.

# Single-GPU (recommended for RTX 4090)
python train_improved.py \
  --batch_size 16 \
  --model_dim 768 \
  --num_layers 12 \
  --seq_length 256 \
  --epochs 20 \
  --num_workers 6 \
  --activation_checkpoint

# Multi-GPU (2 GPUs on single node) - uses torchrun
# torchrun will set the env variables the trainer checks for DDP
# Adjust --nproc_per_node to number of GPUs
# torchrun --nproc_per_node=2 train_improved.py \
#   --batch_size 8 \
#   --model_dim 768 \
#   --num_layers 12 \
#   --seq_length 256 \
#   --epochs 20 \
#   --activation_checkpoint
