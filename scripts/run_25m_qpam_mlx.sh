#!/bin/bash

cd "$(dirname "$0")/.."  
uv run python v6/mlx/train.py \
    --dim 200 --layers 6 --heads 4 --head_dim 16 \
    --batch_size 8 --seq_len 512 --lr 3e-05 --warmup 500 --epochs 10 \
    --save_dir checkpoints_qpam_25m_mlx
