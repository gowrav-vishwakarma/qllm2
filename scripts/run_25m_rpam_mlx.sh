#!/bin/bash
# 25M RPAM training — MLX (real-valued baseline for QPAM comparison)
# Config: dim=344, layers=6, heads=4, head_dim=32, expand=3
# Params: ~25.1M real (matches QPAM 26.8M complex storage)
cd /Users/chris/.npcsh/qllm-private

uv run python v6/mlx/train_real.py \
  --dim 344 \
  --layers 6 \
  --heads 4 \
  --head_dim 32 \
  --batch_size 16 \
  --seq_len 512 \
  --lr 3e-04 \
  --warmup 500 \
  --epochs 10 \
  --save_dir checkpoints_rpam_25m_mlx
