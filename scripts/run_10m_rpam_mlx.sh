#!/bin/bash
# 10M RPAM training — MLX
# Config: dim=128, layers=6, heads=4, head_dim=24, expand=3
# Params: ~10M (matched to 10M QPAM)
cd /Users/caug/npcww/qnlp/qllm-private

uv run python /Users/caug/npcww/qnlp/ket-nlp/qpam_mlx/train_real.py \
  --dim 128 \
  --layers 6 \
  --heads 4 \
  --head_dim 24 \
  --batch_size 16 \
  --seq_len 512 \
  --lr 3e-04 \
  --warmup 500 \
  --epochs 10 \
  --save_dir checkpoints_rpam_10m_mlx
