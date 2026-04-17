#!/bin/bash
# 25M QPAM — matches scaling sweep (lr=1e-4, ket-nlp train.py)
cd /Users/caug/npcww/qnlp/qllm-private
uv run python /Users/caug/npcww/qnlp/ket-nlp/qpam_mlx/train.py \
    --dim 200 --layers 6 --heads 4 --head_dim 16 \
    --batch_size 8 --seq_len 512 --lr 3e-05 --warmup 500 --epochs 10 \
    --save_dir checkpoints_qpam_25m_mlx
