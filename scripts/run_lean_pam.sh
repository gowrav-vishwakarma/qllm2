#!/usr/bin/env bash
# Lean PAM: Stripped-down complex PAM experiment
#
# Architecture: Complex PAM + RoPE + GSP + SimpleComplexFFN (no CGU gating)
# Removes: CGU, hierarchical dt, cross-level drift, multi-scale loss,
#          reverse association, Triton kernels
#
# Two presets:
#   lean_medium_small (~86M, expand=3, same hidden dim as V7 -- default)
#   lean_medium       (~96M, expand=4, param-matched but needs grad ckpt)
#
# Baselines:
#   V7 7a (flat, B=3, ModSwish, CGU): Val PPL 29.73
#   V6 medium-pam-v3:                 Val PPL 29.95
#
# Usage:
#   ./scripts/run_lean_pam.sh
#   ./scripts/run_lean_pam.sh --preset lean_medium_small
#   ./scripts/run_lean_pam.sh --epochs 3      # quick test
#   ./scripts/run_lean_pam.sh --batch_size 6
#   ./scripts/run_lean_pam.sh --resume

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
PRESET="lean_medium_small"
BATCH_SIZE=3
RESUME=0
TAG=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seq_len)    SEQ_LEN="$2";    shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --preset)     PRESET="$2";     shift 2 ;;
        --tag)        TAG="$2";        shift 2 ;;
        --resume)     RESUME=1;        shift ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

GEN_PROMPT="In 1923 , the University of"
RUN_NAME="lean_${PRESET}"
if [[ -n "$TAG" ]]; then
    RUN_NAME="${RUN_NAME}_${TAG}"
fi
CKPT_DIR="checkpoints_${RUN_NAME}"
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"

ARGS="--model lean --preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --no_grad_ckpt"

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
    LOG_DIR=$(make_log_dir "v7" "${RUN_NAME}_${DATASET}")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="Lean PAM ($PRESET): SimpleComplexFFN (no CGU), PAM(complex RoPE, GSP, chunked dual C=256), flat dt=-4.0, LR=1e-4, warmup=1000, B=$BATCH_SIZE"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  Lean PAM: Stripped-down Complex PAM Experiment"
echo "  Preset: $PRESET, B=$BATCH_SIZE"
echo "  Channel mix: SimpleComplexFFN (up -> ModSwish -> down, no gating)"
echo "  Sequence mix: Complex PAM (RoPE, GSP, chunked dual C=256)"
echo "  No: CGU, hierarchical dt, cross-level drift, multi-scale loss,"
echo "      reverse association, Triton kernels"
echo "  seq_len: $SEQ_LEN  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Baseline: 7a val PPL 29.73 (full V7, CGU + PAM)"
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
elapsed=$((end_time - start_time))
echo ""
echo "============================================================"
echo "  Lean PAM complete"
echo "  Wall time: ${elapsed}s ($(printf '%.1f' "$(echo "$elapsed/3600" | bc -l)")h)"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Baseline: 7a val PPL 29.73"
echo "============================================================"
