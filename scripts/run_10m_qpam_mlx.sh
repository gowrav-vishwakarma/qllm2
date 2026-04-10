#!/bin/bash
# 10M QPAM training — MLX
# Config: dim=88, layers=6, heads=4, head_dim=16, expand=3
# Params: ~10M
cd /Users/caug/npcww/qnlp/qllm-private

uv run python /Users/caug/npcww/qnlp/ket-nlp/qpam_mlx/train.py \
  --dim 88 \
  --layers 6 \
  --heads 4 \
  --head_dim 16 \
  --batch_size 16 \
  --seq_len 512 \
  --lr 3e-04 \
  --warmup 500 \
  --epochs 10 \
  --save_dir checkpoints_qpam_10m_mlx
