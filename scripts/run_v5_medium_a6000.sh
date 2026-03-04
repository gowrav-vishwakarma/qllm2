#!/usr/bin/env bash
# Run v5 training on A6000 (48GB), intended for tmux.
#
# Logs to:      logs/v5_train_small-matched.log  (auto, via TeeLogger)
# Checkpoints:  checkpoints_v5/
#
# Suggested workflow:
#   1) Tune batch first (uses same small-matched model):
#      ./scripts/tune_batch_v5_a6000.sh --batch_size 32
#   2) Full run (batch size from step 1 will work):
#      ./scripts/run_v5_medium_a6000.sh --batch_size 32
#   3) Resume if interrupted:
#      ./scripts/run_v5_medium_a6000.sh --resume checkpoints_v5/best_model.pt --epochs 30
#   4) Monitor from another terminal:
#      ./scripts/monitor_training_v5_a6000.sh 5 logs/v5_train_small-matched.log
#
# On server:
#   tmux new -s v5train
#   cd ~/qllm && ./scripts/run_v5_medium_a6000.sh [--batch_size N]
#   # Detach: Ctrl+B, D. Reattach: tmux attach -t v5train

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
  echo "[v5-run] Resuming -- keeping existing checkpoints in $CHECKPOINT_DIR/"
else
  if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
    echo "[v5-run] Fresh start -- clearing old checkpoints in $CHECKPOINT_DIR/"
    rm -rf "$CHECKPOINT_DIR"
  fi
fi

# Uses small-matched to match tune_batch_v5_a6000.sh (so batch size transfers).
# If you want to use 'small' or 'medium' model, tune batch size separately for that size.
eval "$PYTHON_BIN -m v5.train" \
  --size small-matched \
  --max_samples 20000 \
  --seq_len 256 \
  --batch_size 48 \
  --epochs 20 \
  --log_dir logs \
  --checkpoint_dir checkpoints_v5 \
  "$@"

