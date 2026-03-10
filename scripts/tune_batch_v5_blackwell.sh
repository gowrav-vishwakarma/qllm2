#!/usr/bin/env bash
# Quick batch-size tuning for RTX PRO 6000 Blackwell 96GB.
# Defaults to the v5 medium preset with bf16 and a short run so you can
# find the largest stable batch before starting a long compile-enabled job.
#
# Examples:
#   ./scripts/tune_batch_v5_blackwell.sh
#   ./scripts/tune_batch_v5_blackwell.sh --batch_size 24
#   ./scripts/tune_batch_v5_blackwell.sh --size large --batch_size 8

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v5/train.py ]] || cd ..

# Bootstrap env and deps; exports PYTHON_BIN.
# shellcheck disable=SC1091
source ./scripts/v5_env_setup_a6000.sh

CHECKPOINT_DIR="checkpoints_v5_blackwell"
if echo "$@" | grep -q -- '--resume'; then
  echo "[v5-blackwell-tune] Resuming -- keeping existing checkpoints in $CHECKPOINT_DIR/"
else
  if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
    echo "[v5-blackwell-tune] Fresh start -- clearing old checkpoints in $CHECKPOINT_DIR/"
    rm -rf "$CHECKPOINT_DIR"
  fi
fi

eval "$PYTHON_BIN -m v5.train" \
  --size medium \
  --max_samples 20000 \
  --seq_len 256 \
  --batch_size 16 \
  --epochs 1 \
  --init_seed 42 \
  --amp_dtype bf16 \
  --num_workers 8 \
  --log_dir logs \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  "$@"
