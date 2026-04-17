#!/bin/bash
# 25M RPAM — matches scaling sweep (lr=1e-4, ket-nlp train_real.py)
cd /Users/caug/npcww/qnlp/qllm-private
uv run python /Users/caug/npcww/qnlp/ket-nlp/qpam_mlx/train_real.py \
    --dim 344 --layers 6 --heads 4 --head_dim 32 \
    --batch_size 8 --seq_len 512 --lr 3e-05 --warmup 500 --epochs 10 \
    --save_dir checkpoints_rpam_25m_mlx
