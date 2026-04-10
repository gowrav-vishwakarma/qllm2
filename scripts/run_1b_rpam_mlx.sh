#!/bin/bash
# 1B RPAM (real-valued ablation) training — MLX/M4 Max
# Config: dim=1632, layers=24, heads=24, head_dim=68, expand=3
# Params: ~981M (matched to 1B QPAM)
#
# Run: cd qllm-private && bash scripts/run_1b_rpam_mlx.sh

cd /Users/caug/npcww/qnlp/qllm-private

uv run python /Users/caug/npcww/qnlp/ket-nlp/qpam_mlx/train_real.py \
  --dim 1632 \
  --layers 24 \
  --heads 24 \
  --head_dim 68 \
  --batch_size 1 \
  --seq_len 512 \
  --lr 3e-05 \
  --warmup 500 \
  --epochs 10 \
  --save_dir checkpoints_rpam_1b_mlx
