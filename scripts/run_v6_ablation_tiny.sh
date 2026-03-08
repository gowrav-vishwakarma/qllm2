#!/usr/bin/env bash
# V6 Tiny Repetition Ablation -- 5 runs on RTX 4090.
#
# Tests whether repetition is caused by memory-driven memorization
# or is architectural. Each run: tiny config, 100k samples, 5 epochs.
# Estimated: ~5-10 min per run, ~30-50 min total.
#
# All runs share one timestamped+commit directory:
#   logs/v6/ablation_tiny_<YYYYMMDD_HHMMSS>_<commit>[_dirty]/
#     ├── A_baseline/
#     ├── B_no_memory/
#     ├── C_no_wm/
#     ├── D_no_im/
#     └── E_with_attn/
#
# Usage:
#   tmux new -s v6ablation
#   cd ~/Development/qllm2 && bash scripts/run_v6_ablation_tiny.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

COMMON="--size tiny --max_samples 100000 --seq_len 256 --epochs 5 --batch_size 64 --init_seed 42"

GROUP_DIR=$(make_group_prefix "v6" "ablation_tiny")
echo "============================================================"
echo "V6 Tiny Repetition Ablation -- 5 runs"
echo "Group directory: $GROUP_DIR"
echo "============================================================"
echo ""

write_run_info "$GROUP_DIR" "V6 Tiny Ablation Matrix (A-E)" "$COMMON"

# --- Run A: baseline (all memory, no attention) ---
RUN_A="${GROUP_DIR}/A_baseline"
echo ">>> [A] baseline: WM=16, IM=32, attn=off"
write_run_info "$RUN_A" "Ablation A: baseline (WM+IM, no attn)" "$COMMON"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --log_dir "$RUN_A" \
  --checkpoint_dir /tmp/v6_abl_A
echo ""
echo ">>> [A] DONE"
echo ""

# --- Run B: no memory at all ---
RUN_B="${GROUP_DIR}/B_no_memory"
echo ">>> [B] no-memory: WM=0, IM=0, attn=off"
write_run_info "$RUN_B" "Ablation B: no memory (WM=0, IM=0)" "$COMMON --no_working_memory --no_internal_memory"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --no_working_memory --no_internal_memory \
  --log_dir "$RUN_B" \
  --checkpoint_dir /tmp/v6_abl_B
echo ""
echo ">>> [B] DONE"
echo ""

# --- Run C: IM only (no WM) ---
RUN_C="${GROUP_DIR}/C_no_wm"
echo ">>> [C] no-wm: WM=0, IM=32, attn=off"
write_run_info "$RUN_C" "Ablation C: IM only (no WM)" "$COMMON --no_working_memory"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --no_working_memory \
  --log_dir "$RUN_C" \
  --checkpoint_dir /tmp/v6_abl_C
echo ""
echo ">>> [C] DONE"
echo ""

# --- Run D: WM only (no IM) ---
RUN_D="${GROUP_DIR}/D_no_im"
echo ">>> [D] no-im: WM=16, IM=0, attn=off"
write_run_info "$RUN_D" "Ablation D: WM only (no IM)" "$COMMON --no_internal_memory"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --no_internal_memory \
  --log_dir "$RUN_D" \
  --checkpoint_dir /tmp/v6_abl_D
echo ""
echo ">>> [D] DONE"
echo ""

# --- Run E: all memory + attention (last layer) ---
RUN_E="${GROUP_DIR}/E_with_attn"
echo ">>> [E] with-attn: WM=16, IM=32, attn=last layer"
write_run_info "$RUN_E" "Ablation E: all memory + attention" "$COMMON --use_attention --attn_every 0"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --use_attention --attn_every 0 \
  --log_dir "$RUN_E" \
  --checkpoint_dir /tmp/v6_abl_E
echo ""
echo ">>> [E] DONE"
echo ""

echo "============================================================"
echo "All 5 ablation runs complete."
echo "Results in: $GROUP_DIR"
echo "============================================================"
