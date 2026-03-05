#!/usr/bin/env bash
# Run v5 small-matched training.
# Defaults match the full run (100k samples, 10 epochs) from 2026-03-04.
# Batch size tuned per GPU: 16 for RTX 4090 (24GB), 32 for A6000 (48GB).
#
# Logs to:      logs/v5_train_small-matched.log  (auto, via TeeLogger)
# Checkpoints:  checkpoints_v5/
#
# Usage:
#   ./scripts/tune_batch_v5_a6000.sh
#   ./scripts/tune_batch_v5_a6000.sh --batch_size 24
#   ./scripts/tune_batch_v5_a6000.sh --batch_size 32   # A6000
#   ./scripts/tune_batch_v5_a6000.sh --resume checkpoints_v5/best_model.pt

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v5/train.py ]] || cd ..

# Bootstrap env and deps; exports PYTHON_BIN.
# shellcheck disable=SC1091
source ./scripts/v5_env_setup_a6000.sh

# Clean old checkpoints on fresh start, keep them on --resume
CHECKPOINT_DIR="checkpoints_v5"
if echo "$@" | grep -q -- '--resume'; then
  echo "[v5-tune] Resuming -- keeping existing checkpoints in $CHECKPOINT_DIR/"
else
  if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
    echo "[v5-tune] Fresh start -- clearing old checkpoints in $CHECKPOINT_DIR/"
    rm -rf "$CHECKPOINT_DIR"
  fi
fi

# Defaults: full run (100k, 10 epochs), batch 16 for 4090. Use --batch_size 32 on A6000.
# init_seed 42: best orthogonal seed from A/B test (32.77 val PPL at 10 epochs).
eval "$PYTHON_BIN -m v5.train" \
  --size small-matched \
  --max_samples 100000 \
  --seq_len 256 \
  --batch_size 16 \
  --epochs 10 \
  --init_seed 42 \
  --log_dir logs \
  --checkpoint_dir checkpoints_v5 \
  "$@"

