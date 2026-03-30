#!/usr/bin/env bash
# V7 Experiment 7d: Chunked dual form + B=6 (exploit memory savings)
#
# Purpose: Chunked dual form reduces PAM memory from O(T^2) to O(C^2) per chunk.
# This frees enough VRAM to run batch_size=6 WITHOUT gradient checkpointing.
# Exp 3a-A showed B=6 reached val PPL 26.6 (below transformer 27.08!) but
# required grad ckpt -- confounding the comparison. With chunking, we can
# run B=6 cleanly.
#
# Baselines:
#   V7 7c (B=3, chunked):     Val PPL ~29.73 expected, higher tok/s
#   V7 3a-A (B=6, grad ckpt): Val PPL ~26.6 (beat transformer!)
#   Transformer:                Val PPL ~27.08, ~96k tok/s
#
# Usage:
#   ./scripts/run_v7_exp7d_chunked_b6.sh
#   ./scripts/run_v7_exp7d_chunked_b6.sh --epochs 3          # quick test
#   ./scripts/run_v7_exp7d_chunked_b6.sh --batch_size 4      # if OOM with B=6
#   ./scripts/run_v7_exp7d_chunked_b6.sh --resume

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
CHUNK_SIZE=256
RESUME=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seq_len)    SEQ_LEN="$2";    shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --chunk_size) CHUNK_SIZE="$2"; shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --resume)     RESUME=1;        shift ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

GEN_PROMPT="In 1923 , the University of"
CKPT_DIR="checkpoints_v7_exp7d_chunked_b6"
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"

ARGS="--preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --activation swish --chunk_size $CHUNK_SIZE --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --no_grad_ckpt"

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
    LOG_DIR=$(make_log_dir "v7" "exp7d_chunked_b6_${DATASET}")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="V7 Exp7d (Chunked C=$CHUNK_SIZE, B=$BATCH_SIZE): dim=384 L=16 expand=3, PAM(H=6, d=64, flat dt=-4.0, RoPE, fused-QKV, GSP), activation=swish, chunk_size=$CHUNK_SIZE, LR=1e-4, warmup=1000"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  V7 Experiment 7d: Chunked + B=$BATCH_SIZE ($PRESET, ~100M params)"
echo "  16 layers, uniform dt_bias=-4.0, no cross-level drift"
echo "  Activation: ModSwish, Chunk size: $CHUNK_SIZE"
echo "  Architecture: [CGU(expand=3, swish) -> PAM(H=6, d=64, chunked)] x16 + GSP + RoPE"
echo "  dim=384  heads=6  LR=1e-4  warmup=1000"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Dataset: $DATASET"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Target: val PPL < 27 (beat transformer), higher tok/s than 7a"
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
echo "  Experiment 7d complete"
echo "  Wall time: ${elapsed}s ($(printf '%.1f' "$(echo "$elapsed/3600" | bc -l)")h)"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "============================================================"
