#!/usr/bin/env bash
# V6 Full-Dataset Memory Ablation -- 3 runs on RTX 4090.
#
# Tests whether WM/IM cause the PPL-1.2 memorization on full TinyStories.
#
# Run 1: no memory at all     (pure SSM+banks, should match V5 behavior)
# Run 2: tiny-sized memory    (WM=16, IM=32, same as tiny config)
# Run 3: tiny on full dataset (7.3M params, confirms architecture at scale)
#
# Each run does 1 epoch with mid-epoch generation every 5000 batches.
# Estimated: Run 1 ~7h, Run 2 ~7h, Run 3 ~3h = ~17h total.
#
# Usage:
#   tmux new -s v6mem
#   cd ~/Development/qllm2
#   bash scripts/run_v6_fulldata_ablation.sh          # all 3 runs
#   bash scripts/run_v6_fulldata_ablation.sh 1         # run 1 only
#   bash scripts/run_v6_fulldata_ablation.sh 2         # run 2 only
#   bash scripts/run_v6_fulldata_ablation.sh 3         # run 3 only

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

COMMON="--max_samples 9999999 --seq_len 256 --epochs 1 --init_seed 42 --gen_every 5000 --gen_prompt 'Once upon a time, there was a little'"
RUN_FILTER="${1:-all}"

echo "============================================================"
echo "V6 Full-Dataset Memory Ablation"
echo "============================================================"
echo ""

# --- Run 1: small-matched, NO memory ---
if [[ "$RUN_FILTER" == "all" || "$RUN_FILTER" == "1" ]]; then
  echo ">>> [1] small-matched, NO memory (WM=0, IM=0)"
  RUN1_DIR=$(make_log_dir "v6" "fulldata_no_memory")
  write_run_info "$RUN1_DIR" "Full-data ablation: no memory" \
    "--size small-matched $COMMON --no_working_memory --no_internal_memory --batch_size 20"
  eval "$PYTHON_BIN -m v6.train" \
    --size small-matched $COMMON \
    --no_working_memory --no_internal_memory \
    --batch_size 20 \
    --log_dir "$RUN1_DIR" \
    --checkpoint_dir /tmp/v6_fulldata_no_mem
  echo ""
  echo ">>> [1] DONE"
  echo ""
fi

# --- Run 2: small-matched, tiny-sized memory (WM=16, IM=32) ---
if [[ "$RUN_FILTER" == "all" || "$RUN_FILTER" == "2" ]]; then
  echo ">>> [2] small-matched, tiny memory (WM=16, IM=32)"
  RUN2_DIR=$(make_log_dir "v6" "fulldata_tiny_memory")
  write_run_info "$RUN2_DIR" "Full-data ablation: tiny memory (WM=16, IM=32)" \
    "--size small-matched $COMMON --wm_slots 16 --im_slots 32 --batch_size 20"
  eval "$PYTHON_BIN -m v6.train" \
    --size small-matched $COMMON \
    --wm_slots 16 --im_slots 32 \
    --batch_size 20 \
    --log_dir "$RUN2_DIR" \
    --checkpoint_dir /tmp/v6_fulldata_tiny_mem
  echo ""
  echo ">>> [2] DONE"
  echo ""
fi

# --- Run 3: tiny model on full dataset ---
if [[ "$RUN_FILTER" == "all" || "$RUN_FILTER" == "3" ]]; then
  echo ">>> [3] tiny model (7.3M params) on full dataset"
  RUN3_DIR=$(make_log_dir "v6" "fulldata_tiny_model")
  write_run_info "$RUN3_DIR" "Full-data ablation: tiny model" \
    "--size tiny $COMMON --batch_size 64"
  eval "$PYTHON_BIN -m v6.train" \
    --size tiny $COMMON \
    --batch_size 64 \
    --log_dir "$RUN3_DIR" \
    --checkpoint_dir /tmp/v6_fulldata_tiny_model
  echo ""
  echo ">>> [3] DONE"
  echo ""
fi

echo "============================================================"
echo "Ablation complete."
echo "============================================================"
