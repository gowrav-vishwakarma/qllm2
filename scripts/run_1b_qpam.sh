#!/bin/bash
# 1B QPAM training — PyTorch/CUDA
# Run on GPU machine: cd qllm-private && bash scripts/run_1b_qpam.sh
#
# Config: dim=1152, layers=24, heads=16, head_dim=72, expand=3
# Params: ~979M
# Dataset: WikiText-103

python -m v6.train \
  --size medium-pam-v3-1b \
  --dataset wikitext103 \
  --epochs 10 \
  --batch_size 2 \
  --seq_len 512 \
  --lr 3e-5 \
  --lr_schedule warmup_cosine \
  --warmup_steps 500 \
  --compile \
  --amp_dtype bf16
