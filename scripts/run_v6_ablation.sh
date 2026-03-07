#!/usr/bin/env bash
# Run V6 ablation experiments.
#
# Tests the contribution of working memory and internal memory
# by training matched models with components disabled.
#
# Usage:
#   ./scripts/run_v6_ablation.sh                     # runs all 3 configs
#   ./scripts/run_v6_ablation.sh --max_samples 5000  # quick test

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh

MAX_SAMPLES="${1:-100000}"
EPOCHS=10
BATCH=16
SEQ=256

# Override with any passed args
for arg in "$@"; do
  case "$arg" in
    --max_samples=*) MAX_SAMPLES="${arg#*=}" ;;
    --epochs=*) EPOCHS="${arg#*=}" ;;
    --batch_size=*) BATCH="${arg#*=}" ;;
  esac
done

echo "============================================================"
echo "V6 Ablation Study"
echo "============================================================"
echo "Samples: $MAX_SAMPLES, Epochs: $EPOCHS, Batch: $BATCH, Seq: $SEQ"
echo ""

# Run A: Full model (baseline)
echo "=== Run A: Full V6 (working memory + internal memory) ==="
eval "$PYTHON_BIN -m v6.train" \
  --size small-matched \
  --max_samples "$MAX_SAMPLES" --seq_len "$SEQ" --batch_size "$BATCH" --epochs "$EPOCHS" \
  --init_seed 42 \
  --log_dir logs/v6_ablation \
  --checkpoint_dir checkpoints_v6_ablation/full

# Run B: No working memory
echo ""
echo "=== Run B: No working memory ==="
eval "$PYTHON_BIN -m v6.train" \
  --size small-matched \
  --max_samples "$MAX_SAMPLES" --seq_len "$SEQ" --batch_size "$BATCH" --epochs "$EPOCHS" \
  --init_seed 42 --no_working_memory \
  --log_dir logs/v6_ablation \
  --checkpoint_dir checkpoints_v6_ablation/no_wm

# Run C: No internal memory
echo ""
echo "=== Run C: No internal memory ==="
eval "$PYTHON_BIN -m v6.train" \
  --size small-matched \
  --max_samples "$MAX_SAMPLES" --seq_len "$SEQ" --batch_size "$BATCH" --epochs "$EPOCHS" \
  --init_seed 42 --no_internal_memory \
  --log_dir logs/v6_ablation \
  --checkpoint_dir checkpoints_v6_ablation/no_im

echo ""
echo "============================================================"
echo "Ablation complete. Compare logs in logs/v6_ablation/"
echo "============================================================"
