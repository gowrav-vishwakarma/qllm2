#!/bin/bash
# 25M QPAM training — MLX
# Config: dim=200, layers=6, heads=4, head_dim=16, expand=3
# Params: ~25M
cd /Users/chris/.npcsh/qllm-private

uv run python v6/mlx/train.py \
  --dim 200 \
  --layers 6 \
  --heads 4 \
  --head_dim 16 \
  --batch_size 16 \
  --seq_len 512 \
  --lr 3e-04 \
  --warmup 500 \
  --epochs 10 \
  --save_dir checkpoints_qpam_25m_mlx
