#!/usr/bin/env bash
# V6 Tiny Repetition Ablation -- 5 runs on RTX 4090.
#
# Tests whether repetition is caused by memory-driven memorization
# or is architectural. Each run: tiny config, 100k samples, 5 epochs.
# Estimated: ~5-10 min per run, ~30-50 min total.
#
# Usage:
#   tmux new -s v6ablation
#   cd ~/Development/qllm2 && bash scripts/run_v6_ablation_tiny.sh
#
# Results go to logs/v6_ablation_tiny_{A..E}/ directories.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh

COMMON="--size tiny --max_samples 100000 --seq_len 256 --epochs 5 --batch_size 64 --init_seed 42"

echo "============================================================"
echo "V6 Tiny Repetition Ablation -- 5 runs"
echo "============================================================"
echo ""

# --- Run A: baseline (all memory, no attention) ---
echo ">>> [A] baseline: WM=16, IM=32, attn=off"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --log_dir logs/v6_ablation_tiny_A_baseline \
  --checkpoint_dir /tmp/v6_abl_A
echo ""
echo ">>> [A] DONE"
echo ""

# --- Run B: no memory at all ---
echo ">>> [B] no-memory: WM=0, IM=0, attn=off"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --no_working_memory --no_internal_memory \
  --log_dir logs/v6_ablation_tiny_B_no_memory \
  --checkpoint_dir /tmp/v6_abl_B
echo ""
echo ">>> [B] DONE"
echo ""

# --- Run C: IM only (no WM) ---
echo ">>> [C] no-wm: WM=0, IM=32, attn=off"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --no_working_memory \
  --log_dir logs/v6_ablation_tiny_C_no_wm \
  --checkpoint_dir /tmp/v6_abl_C
echo ""
echo ">>> [C] DONE"
echo ""

# --- Run D: WM only (no IM) ---
echo ">>> [D] no-im: WM=16, IM=0, attn=off"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --no_internal_memory \
  --log_dir logs/v6_ablation_tiny_D_no_im \
  --checkpoint_dir /tmp/v6_abl_D
echo ""
echo ">>> [D] DONE"
echo ""

# --- Run E: all memory + attention (last layer) ---
echo ">>> [E] with-attn: WM=16, IM=32, attn=last layer"
eval "$PYTHON_BIN -m v6.train" $COMMON \
  --use_attention --attn_every 0 \
  --log_dir logs/v6_ablation_tiny_E_with_attn \
  --checkpoint_dir /tmp/v6_abl_E
echo ""
echo ">>> [E] DONE"
echo ""

echo "============================================================"
echo "All 5 ablation runs complete."
echo "Logs in: logs/v6_ablation_tiny_{A..E}/"
echo "============================================================"
