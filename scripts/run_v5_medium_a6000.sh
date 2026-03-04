#!/usr/bin/env bash
# Run v5 training on A6000 (48GB), intended for tmux.
#
# Suggested workflow:
#   1) Tune batch first:
#      ./scripts/tune_batch_v5_a6000.sh --batch_size 32
#   2) Full run:
#      ./scripts/run_v5_medium_a6000.sh --batch_size 32
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

mkdir -p logs checkpoints_v5 2>/dev/null || true

# Default run uses small config (dim=256, 8 layers) with benchmark-sized subset.
eval "$PYTHON_BIN -m v5.train" \
  --size small \
  --max_samples 20000 \
  --seq_len 256 \
  --batch_size 48 \
  --epochs 20 \
  "$@"

