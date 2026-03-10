#!/usr/bin/env bash
# Next small-model Blackwell experiment:
# - warmup + cosine
# - slightly stronger dropout
# - slightly stronger weight decay

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v5/train.py ]] || cd ..

exec ./scripts/run_v5_blackwell.sh \
  --size small \
  --batch_size 64 \
  --seq_len 512 \
  --window_size 512 \
  --attention_backend xformers \
  --lr_schedule warmup_cosine \
  --warmup_steps 2000 \
  --dropout 0.15 \
  --weight_decay 0.03 \
  "$@"
