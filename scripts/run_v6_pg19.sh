#!/usr/bin/env bash
# Run V6 training on PG-19 (Project Gutenberg books) -- RTX 4090.
#
# Tests multi-timescale SSM slow lanes with book-length character persistence.
# Run after WikiText-103 validates the core architecture.
#
# Usage:
#   ./scripts/run_v6_pg19.sh                              # baseline (no memory)
#   ./scripts/run_v6_pg19.sh --wm_slots 64 --im_slots 128 # with memory

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

TRAIN_ARGS="--dataset pg19 --size small-matched --max_samples 9999999 --seq_len 1024 --batch_size 12 --epochs 5 --init_seed 42 --gen_every 5000 --gen_prompt 'It was a dark and stormy night when' --compile --compile_mode reduce-overhead --amp_dtype auto --num_workers 4"

LOG_DIR=$(make_log_dir "v6" "pg19_small_matched")
echo "[v6-run] Log directory: $LOG_DIR"

write_run_info "$LOG_DIR" "V6 small-matched on PG-19 books (RTX 4090)" "$TRAIN_ARGS $*"

CHECKPOINT_DIR="checkpoints_v6_pg19"
if echo "$@" | grep -q -- '--resume'; then
  echo "[v6-run] Resuming -- keeping existing checkpoints in $CHECKPOINT_DIR/"
else
  if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
    echo "[v6-run] Fresh start -- clearing old checkpoints in $CHECKPOINT_DIR/"
    rm -rf "$CHECKPOINT_DIR"
  fi
fi

eval "$PYTHON_BIN -m v6.train" \
  $TRAIN_ARGS \
  --log_dir "$LOG_DIR" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  "$@"
