#!/usr/bin/env bash
# Find best batch size for v5 on A6000.
# Runs WITHOUT compile so each try is quick.
#
# Usage:
#   ./scripts/tune_batch_v5_a6000.sh
#   ./scripts/tune_batch_v5_a6000.sh --batch_size 24
#   ./scripts/tune_batch_v5_a6000.sh --batch_size 32 --epochs 2

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v5/train.py ]] || cd ..

# Bootstrap env and deps; exports PYTHON_BIN.
# shellcheck disable=SC1091
source ./scripts/v5_env_setup_a6000.sh

mkdir -p logs checkpoints_v5 2>/dev/null || true

# Default is small-matched so comparison with small baselines is fairer.
eval "$PYTHON_BIN -m v5.train" \
  --size small-matched \
  --max_samples 20000 \
  --seq_len 256 \
  --batch_size 64 \
  --epochs 5 \
  "$@"

