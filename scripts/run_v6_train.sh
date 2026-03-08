#!/usr/bin/env bash
# Run V6 training on GPU (RTX 4090 / A6000), intended for tmux.
#
# Logs go to:   logs/v6/small_matched_<timestamp>_<commit>[_dirty]/
# Checkpoints:  checkpoints_v6/
#
# Suggested workflow:
#   1) Tune batch first:
#      ./scripts/tune_batch_v6.sh --batch_size 16
#   2) Full run:
#      ./scripts/run_v6_train.sh --batch_size 16
#   3) Resume if interrupted:
#      ./scripts/run_v6_train.sh --resume checkpoints_v6/best_model.pt --epochs 20
#   4) Monitor from another terminal:
#      ./scripts/monitor_v6.sh 5 <log_dir>/v6_train_small-matched.log
#
# Ablation runs:
#   ./scripts/run_v6_train.sh --no_working_memory
#   ./scripts/run_v6_train.sh --no_internal_memory
#
# On server:
#   tmux new -s v6train
#   cd ~/qllm && ./scripts/run_v6_train.sh [--batch_size N]
#   # Detach: Ctrl+B, D. Reattach: tmux attach -t v6train

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

TRAIN_ARGS="--size small-matched --max_samples 100000 --seq_len 256 --batch_size 16 --epochs 10 --init_seed 42"

LOG_DIR=$(make_log_dir "v6" "small_matched")
echo "[v6-run] Log directory: $LOG_DIR"

write_run_info "$LOG_DIR" "V6 small-matched training" "$TRAIN_ARGS $*"

CHECKPOINT_DIR="checkpoints_v6"
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
