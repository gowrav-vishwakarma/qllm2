#!/usr/bin/env bash
# Run V6 ablation experiments (small-matched size).
#
# Tests the contribution of working memory and internal memory
# by training matched models with components disabled.
#
# All runs share one timestamped+commit directory:
#   logs/v6/ablation_<YYYYMMDD_HHMMSS>_<commit>[_dirty]/
#     ├── A_full/
#     ├── B_no_wm/
#     └── C_no_im/
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
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

MAX_SAMPLES="${1:-100000}"
EPOCHS=10
BATCH=16
SEQ=256

for arg in "$@"; do
  case "$arg" in
    --max_samples=*) MAX_SAMPLES="${arg#*=}" ;;
    --epochs=*) EPOCHS="${arg#*=}" ;;
    --batch_size=*) BATCH="${arg#*=}" ;;
  esac
done

COMMON="--size small-matched --max_samples $MAX_SAMPLES --seq_len $SEQ --batch_size $BATCH --epochs $EPOCHS --init_seed 42"

GROUP_DIR=$(make_group_prefix "v6" "ablation")
echo "============================================================"
echo "V6 Ablation Study"
echo "Group directory: $GROUP_DIR"
echo "============================================================"
echo "Samples: $MAX_SAMPLES, Epochs: $EPOCHS, Batch: $BATCH, Seq: $SEQ"
echo ""

write_run_info "$GROUP_DIR" "V6 Ablation Study (A-C, small-matched)" "$COMMON"

# Run A: Full model (baseline)
RUN_A="${GROUP_DIR}/A_full"
echo "=== Run A: Full V6 (working memory + internal memory) ==="
write_run_info "$RUN_A" "Ablation A: full model (WM+IM)" "$COMMON"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --log_dir "$RUN_A" \
  --checkpoint_dir checkpoints_v6_ablation/full

# Run B: No working memory
RUN_B="${GROUP_DIR}/B_no_wm"
echo ""
echo "=== Run B: No working memory ==="
write_run_info "$RUN_B" "Ablation B: no WM" "$COMMON --no_working_memory"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --no_working_memory \
  --log_dir "$RUN_B" \
  --checkpoint_dir checkpoints_v6_ablation/no_wm

# Run C: No internal memory
RUN_C="${GROUP_DIR}/C_no_im"
echo ""
echo "=== Run C: No internal memory ==="
write_run_info "$RUN_C" "Ablation C: no IM" "$COMMON --no_internal_memory"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --no_internal_memory \
  --log_dir "$RUN_C" \
  --checkpoint_dir checkpoints_v6_ablation/no_im

echo ""
echo "============================================================"
echo "Ablation complete. Results in: $GROUP_DIR"
echo "============================================================"
