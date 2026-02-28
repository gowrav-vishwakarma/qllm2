#!/usr/bin/env bash
# Find best batch size for A6000: runs WITHOUT compile so each batch-size try is fast.
# After 1 epoch (or when you see stable GPU use), kill (Ctrl+C) and note the batch size.
# Then run full training with compile: ./scripts/run_v4_medium_a6000.sh --batch_size N
#
# Usage:
#   ./scripts/tune_batch_a6000.sh                    # default batch_size 16, 1 epoch
#   ./scripts/tune_batch_a6000.sh --batch_size 24   # try 24
#   ./scripts/tune_batch_a6000.sh --batch_size 32 --epochs 2  # 2 epochs

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v4/train_real.py ]] || cd ..
export PATH="$HOME/.local/bin:$PATH"

mkdir -p .cache/v4_tokens checkpoints_v4_real
mkdir -p logs 2>/dev/null || true

# No --compile: fast iterations. Default 1 epoch so you can kill after one full epoch.
uv run python v4/train_real.py \
  --dataset tinystories \
  --size medium \
  --max_length 256 \
  --batch_size 64 \
  --accumulation_steps 1 \
  --epochs 10 \
  --num_workers 4 \
  --no_metrics \
  --cache_dir .cache/v4_tokens \
  --checkpoint_dir checkpoints_v4_real \
  "$@"
