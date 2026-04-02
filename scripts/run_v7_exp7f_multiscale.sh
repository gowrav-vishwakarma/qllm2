#!/usr/bin/env bash
# V7 Experiment 7f: Multi-scale per-layer loss (no reverse association)
#
# Purpose: Each layer gets an auxiliary prediction head that predicts tokens at a
# temporal offset matching its memory span. Global layers predict far ahead (t+32),
# step layers predict next-token (t+1). This gives every layer a direct gradient
# signal at its natural timescale, encouraging early layers to encode facts and
# later layers to handle syntax/grammar.
#
# Uses grouped hierarchy (medium_h16_grouped) so dt_bias varies by layer group.
# Reverse association is disabled to isolate the multi-scale loss effect.
#
# Baselines:
#   V7 7d (flat, chunked, B=6): Val PPL ~27.94, 31.8k tok/s
#
# Usage:
#   ./scripts/run_v7_exp7f_multiscale.sh
#   ./scripts/run_v7_exp7f_multiscale.sh --epochs 3          # quick test
#   ./scripts/run_v7_exp7f_multiscale.sh --batch_size 4      # if OOM
#   ./scripts/run_v7_exp7f_multiscale.sh --resume

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
PRESET="medium_h16_grouped"
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
CKPT_DIR="checkpoints_v7_exp7f_multiscale"
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
    LOG_DIR=$(make_log_dir "v7" "exp7f_multiscale_${DATASET}")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="V7 Exp7f (Multi-scale loss, grouped hierarchy, no rev_assoc): dim=384 L=16 expand=3, PAM(H=6, d=64, grouped dt, RoPE, fused-QKV, GSP), activation=swish, chunk_size=$CHUNK_SIZE, aux_weight=$AUX_LOSS_WEIGHT, aux_stride=$AUX_LAYER_STRIDE, max_offset=$MAX_AUX_OFFSET, LR=1e-4, warmup=1000"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  V7 Experiment 7f: Multi-Scale Loss ($PRESET, ~100M params)"
echo "  16 layers, grouped hierarchy, cross-level drift"
echo "  Activation: ModSwish, Chunk size: $CHUNK_SIZE"
echo "  Multi-scale: aux_weight=$AUX_LOSS_WEIGHT stride=$AUX_LAYER_STRIDE max_offset=$MAX_AUX_OFFSET"
echo "  Reverse association: DISABLED (isolate multi-scale effect)"
echo "  Architecture: [CGU(expand=3, swish) -> PAM(H=6, d=64, chunked)] x16 + GSP + RoPE"
echo "  dim=384  heads=6  LR=1e-4  warmup=1000"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Dataset: $DATASET"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Target: val PPL < 27.94 (beat 7d baseline)"
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
echo "  Experiment 7f complete"
echo "  Wall time: ${elapsed}s ($(printf '%.1f' "$(echo "$elapsed/3600" | bc -l)")h)"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "============================================================"
