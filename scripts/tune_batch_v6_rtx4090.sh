#!/usr/bin/env bash
# Tune batch size for V6 on RTX 4090 (24GB).
# Quick 1-epoch run to find max batch before OOM.
#
# Usage:
#   ./scripts/tune_batch_v6_rtx4090.sh                    # try default batch 16
#   ./scripts/tune_batch_v6_rtx4090.sh --batch_size 24    # try 24
#   ./scripts/tune_batch_v6_rtx4090.sh --batch_size 32    # try 32 (may OOM)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh

CHECKPOINT_DIR="checkpoints_v6_tune"
rm -rf "$CHECKPOINT_DIR" 2>/dev/null

eval "$PYTHON_BIN -m v6.train" \
  --size small-matched \
  --max_samples 5000 \
  --seq_len 256 \
  --batch_size 16 \
  --epochs 1 \
  --log_dir logs \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  "$@"

echo ""
echo "[v6-tune] Done. If no OOM, this batch size works on RTX 4090."
echo "[v6-tune] Try increasing --batch_size for higher throughput."
