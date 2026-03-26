#!/usr/bin/env bash
# V7 Experiment 3a: Depth-matched flat baseline (~100M params) on WikiText-103
#
# Purpose: Verify V7 codebase matches V6 medium-pam-v3 at identical depth/width.
# Shape: 16 layers, dim=384, H=6, head_dim=64, expand=3 — same as V6 pam-v3.
# No hierarchy (uniform dt_bias=-4.0), no cross-level drift.
#
# Target: Val PPL ~30 (matching V6 medium-pam-v3's 29.95)
#
# Baselines:
#   V6 medium-pam-v3 (100.4M, 16L, same shape): Val PPL 29.95
#   Transformer      (100.3M, 12L):              Val PPL ~27
#   V7 medium_h6     (105M, 6L wide):            Val PPL 98.3 (3 epochs)
#
# Usage:
#   ./scripts/run_v7_medium_h16_flat.sh                     # full run
#   ./scripts/run_v7_medium_h16_flat.sh --epochs 3          # quick test
#   ./scripts/run_v7_medium_h16_flat.sh --batch_size 2      # if OOM
#   ./scripts/run_v7_medium_h16_flat.sh --resume            # resume from checkpoint
#   ./scripts/run_v7_medium_h16_flat.sh --preset medium_h16_grouped  # run 3b instead

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v7/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=10
SEQ_LEN=2048
DATASET="wikitext103"
PRESET="medium_h16_flat"
BATCH_SIZE=6
RESUME=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seq_len)    SEQ_LEN="$2";    shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --preset)     PRESET="$2";     shift 2 ;;
        --resume)     RESUME=1;        shift ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

GEN_PROMPT="In 1923 , the University of"
CKPT_DIR="checkpoints_v7_${PRESET}"
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"

ARGS="--preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000"

RESUME_ARG=""
REUSED_LOG_DIR=0
LOG_DIR=""

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" && -f "$LOG_DIR_SIDECAR" ]]; then
    _stored=$(head -n 1 "$LOG_DIR_SIDECAR" | tr -d '\r')
    if [[ -n "$_stored" && -d "$_stored" ]]; then
        LOG_DIR="$_stored"
        REUSED_LOG_DIR=1
        echo "[resume] Reusing log directory from $LOG_DIR_SIDECAR: $LOG_DIR"
    elif [[ -n "$_stored" ]]; then
        echo "[resume] Warning: stored log dir not found on disk: $_stored (will create a new log dir)" >&2
    fi
fi

if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR=$(make_log_dir "v7" "${PRESET}_${DATASET}")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="V7 Exp3a ($PRESET): dim=384 L=16 expand=3, PAM(H=6, d=64, flat dt=-4.0, RoPE, fused-QKV, GSP), no hierarchy, no cross-level, LR=1e-4, warmup=1000"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  V7 Experiment 3a: Flat Baseline ($PRESET, ~100M params)"
echo "  16 layers, uniform dt_bias=-4.0, no cross-level drift"
echo "  Architecture: [CGU(expand=3) -> PAM(H=6, d=64)] x16 + GSP + RoPE"
echo "  dim=384  heads=6  LR=1e-4  warmup=1000"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Dataset: $DATASET"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Target: Val PPL ~30 (match V6 medium-pam-v3)"
echo "============================================================"
echo ""

if [[ $REUSED_LOG_DIR -eq 1 ]]; then
    append_run_info_resume "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"
else
    write_run_info "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"
fi

start_time=$(date +%s)

eval "$PYTHON_BIN -m v7.train" \
    $ARGS \
    --gen_prompt "'$GEN_PROMPT'" \
    --log_dir "$LOG_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    $RESUME_ARG \
    $EXTRA_ARGS

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
mins=$(( elapsed / 60 ))
secs=$(( elapsed % 60 ))

echo ""
echo "============================================================"
echo "  V7 Experiment 3a complete!"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo ""
echo "  Baseline comparisons:"
echo "    V6 medium-pam-v3 (100.4M, 16L, same shape): Val PPL 29.95"
echo "    Transformer      (100.3M, 12L):              Val PPL ~27"
echo "    V7 medium_h6     (105M, 6L wide):            Val PPL 98.3"
echo "============================================================"
