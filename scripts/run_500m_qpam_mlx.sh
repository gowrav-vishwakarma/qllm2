#!/bin/bash
# 500M QPAM training — MLX (Apple Silicon)
# Config: dim=576, layers=24, heads=8, head_dim=72, expand=3
# Estimated params: ~500M

# IMPORTANT: Make sure mlx is installed in your uv environment:
# uv add mlx mlx-nn mlx-optimizers

cd /Users/chris/.npcsh/qllm-private

# Run the QPAM MLX training with 500M config
python -m v6.mlx.train   --dim 576   --layers 24   --heads 8   --head_dim 72   --expand 3   --batch_size 2   --seq_len 512   --lr 1e-4   --warmup 2000   --epochs 10   --log_every 50   --val_every 500   --max_train 20000   --max_val 1000   --save_dir checkpoints_qpam_500m_mlx
