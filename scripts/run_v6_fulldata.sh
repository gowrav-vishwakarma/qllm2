#!/usr/bin/env bash
# Run V6 on full TinyStories dataset (2.1M texts, ~474M tokens).
# Expect ~7h/epoch on RTX 4090, ~103K batches/epoch at batch=16.
#
# Logs go to:   logs/v6/fulldata_<timestamp>_<commit>[_dirty]/
# Checkpoints:  checkpoints_v6_full/
#
# Usage:
#   ./scripts/run_v6_fulldata.sh
#   ./scripts/run_v6_fulldata.sh --batch_size 32    # A6000
#   ./scripts/run_v6_fulldata.sh --resume checkpoints_v6_full/best_model.pt
#
# On server:
#   tmux new -s v6full
#   cd ~/qllm && ./scripts/run_v6_fulldata.sh
#   # Detach: Ctrl+B, D. Reattach: tmux attach -t v6full

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

TRAIN_ARGS="--size small-matched --max_samples 9999999 --seq_len 256 --batch_size 16 --epochs 10 --init_seed 42"

LOG_DIR=$(make_log_dir "v6" "fulldata")
echo "[v6-full] Log directory: $LOG_DIR"

write_run_info "$LOG_DIR" "V6 full TinyStories training" "$TRAIN_ARGS $*"

CHECKPOINT_DIR="checkpoints_v6_full"
if echo "$@" | grep -q -- '--resume'; then
  echo "[v6-full] Resuming -- keeping existing checkpoints in $CHECKPOINT_DIR/"
else
  if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
    echo "[v6-full] Fresh start -- clearing old checkpoints in $CHECKPOINT_DIR/"
    rm -rf "$CHECKPOINT_DIR"
  fi
fi

eval "$PYTHON_BIN -m v6.train" \
  $TRAIN_ARGS \
  --log_dir "$LOG_DIR" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  "$@"
