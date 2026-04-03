#!/usr/bin/env bash
# V7 Experiment 7f-2: Reverse association ONLY on flat baseline
#
# Ablation: Changes exactly ONE variable vs 7a baseline.
#   7a baseline: medium_h16_flat, B=3, ModSwish, chunk_size=256, no grad ckpt
#   This run:    + reverse_assoc (reuse upper triangle of Q@K.T, learnable rev_scale init=0)
#
# Multi-scale loss is DISABLED to isolate the reverse association effect.
# Grouped hierarchy is NOT used (stays flat like 7a).
#
# Expected overhead: ~25% more FLOPs in dual form block (one extra [T,T]@[T,d]
# matmul per chunk), zero extra Q/K/V matmuls. rev_scale starts at 0.0 so the
# model discovers the reverse signal via gradient descent.
#
# Baselines:
#   V7 7a (flat, B=3, ModSwish): Val PPL 29.73
#
# Usage:
#   ./scripts/run_v7_exp7f2_reverse_assoc.sh
#   ./scripts/run_v7_exp7f2_reverse_assoc.sh --epochs 3      # quick test
#   ./scripts/run_v7_exp7f2_reverse_assoc.sh --resume

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
CKPT_DIR="checkpoints_v7_exp7f2_reverse_assoc"
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"

# reverse_assoc=True by default in V7Config; no --no_reverse_assoc flag needed
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
    LOG_DIR=$(make_log_dir "v7" "exp7f2_reverse_assoc_${DATASET}")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="V7 Exp7f-2 (Reverse assoc, flat baseline): dim=384 L=16 expand=3, PAM(H=6, d=64, flat dt=-4.0, RoPE, fused-QKV, GSP, rev_assoc=True), activation=swish, chunk_size=$CHUNK_SIZE, LR=1e-4, warmup=1000"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  V7 Experiment 7f-2: Reverse Association on Flat Baseline"
echo "  Ablation: reverse_assoc ONLY (vs 7a baseline)"
echo "  Preset: $PRESET, ~100M params, B=$BATCH_SIZE"
echo "  Reverse association: ENABLED (rev_scale per layer, init=0)"
echo "  Multi-scale loss: DISABLED"
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
echo "  Experiment 7f-2 complete"
echo "  Wall time: ${elapsed}s ($(printf '%.1f' "$(echo "$elapsed/3600" | bc -l)")h)"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Baseline: 7a val PPL 29.73"
echo "============================================================"
