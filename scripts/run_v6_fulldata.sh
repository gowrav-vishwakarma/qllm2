#!/usr/bin/env bash
# Run V6 on full TinyStories dataset (2.1M texts, ~474M tokens).
# Expect ~7h/epoch on RTX 4090, ~103K batches/epoch at batch=16.
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
  --size small-matched \
  --max_samples 9999999 \
  --seq_len 256 \
  --batch_size 16 \
  --epochs 10 \
  --init_seed 42 \
  --log_dir logs \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  "$@"
