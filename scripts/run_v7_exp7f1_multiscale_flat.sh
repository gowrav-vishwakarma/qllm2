#!/usr/bin/env bash
# V7 Experiment 7f-1: Multi-scale loss ONLY on flat baseline
#
# Ablation: Changes exactly ONE variable vs 7a baseline.
#   7a baseline: medium_h16_flat, B=3, ModSwish, chunk_size=256, no grad ckpt
#   This run:    + multi_scale_loss (aux heads every 3 layers, offsets up to t+32)
#
# Reverse association is DISABLED to isolate the multi-scale loss effect.
# Grouped hierarchy is NOT used (stays flat like 7a).
#
# Baselines:
#   V7 7a (flat, B=3, ModSwish): Val PPL 29.73
#
# Usage:
#   ./scripts/run_v7_exp7f1_multiscale_flat.sh
#   ./scripts/run_v7_exp7f1_multiscale_flat.sh --epochs 3      # quick test
#   ./scripts/run_v7_exp7f1_multiscale_flat.sh --resume

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
BATCH_SIZE=3
CHUNK_SIZE=256
AUX_LOSS_WEIGHT=0.1
AUX_LAYER_STRIDE=3
MAX_AUX_OFFSET=32
RESUME=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)           EPOCHS="$2";           shift 2 ;;
        --seq_len)          SEQ_LEN="$2";          shift 2 ;;
        --batch_size)       BATCH_SIZE="$2";       shift 2 ;;
        --chunk_size)       CHUNK_SIZE="$2";       shift 2 ;;
        --dataset)          DATASET="$2";          shift 2 ;;
        --aux_loss_weight)  AUX_LOSS_WEIGHT="$2";  shift 2 ;;
        --aux_layer_stride) AUX_LAYER_STRIDE="$2"; shift 2 ;;
        --max_aux_offset)   MAX_AUX_OFFSET="$2";   shift 2 ;;
        --resume)           RESUME=1;              shift ;;
        *)                  EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

GEN_PROMPT="In 1923 , the University of"
CKPT_DIR="checkpoints_v7_exp7f1_multiscale_flat"
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"

ARGS="--preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --activation swish --chunk_size $CHUNK_SIZE --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --no_grad_ckpt --multi_scale_loss --aux_loss_weight $AUX_LOSS_WEIGHT --aux_layer_stride $AUX_LAYER_STRIDE --max_aux_offset $MAX_AUX_OFFSET --no_reverse_assoc"

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
    LOG_DIR=$(make_log_dir "v7" "exp7f1_multiscale_flat_${DATASET}")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="V7 Exp7f-1 (Multi-scale loss, flat baseline): dim=384 L=16 expand=3, PAM(H=6, d=64, flat dt=-4.0, RoPE, fused-QKV, GSP), activation=swish, chunk_size=$CHUNK_SIZE, aux_weight=$AUX_LOSS_WEIGHT, aux_stride=$AUX_LAYER_STRIDE, max_offset=$MAX_AUX_OFFSET, rev_assoc=False, LR=1e-4, warmup=1000"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  V7 Experiment 7f-1: Multi-Scale Loss on Flat Baseline"
echo "  Ablation: multi_scale_loss ONLY (vs 7a baseline)"
echo "  Preset: $PRESET, ~100M params, B=$BATCH_SIZE"
echo "  Multi-scale: aux_weight=$AUX_LOSS_WEIGHT stride=$AUX_LAYER_STRIDE max_offset=$MAX_AUX_OFFSET"
echo "  Reverse association: DISABLED"
echo "  Grouped hierarchy: NO (flat, same as 7a)"
echo "  seq_len: $SEQ_LEN  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Baseline: 7a val PPL 29.73"
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
echo "  Experiment 7f-1 complete"
echo "  Wall time: ${elapsed}s ($(printf '%.1f' "$(echo "$elapsed/3600" | bc -l)")h)"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Baseline: 7a val PPL 29.73"
echo "============================================================"
