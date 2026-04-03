#!/usr/bin/env bash
# V7 Experiment 7f-3: Grouped hierarchy ONLY (no new features)
#
# Ablation: Changes exactly ONE variable vs 7a baseline.
#   7a baseline: medium_h16_flat (uniform dt_bias=-4.0, no cross_level)
#   This run:    medium_h16_grouped (grouped dt_bias schedule, cross_level=True)
#
# Multi-scale loss and reverse association are DISABLED to isolate the grouped
# hierarchy effect. This tells us whether the dt_bias grouping and cross-level
# drift help or hurt PPL on their own.
#
# Baselines:
#   V7 7a (flat, B=3, ModSwish): Val PPL 29.73
#
# Usage:
#   ./scripts/run_v7_exp7f3_grouped_only.sh
#   ./scripts/run_v7_exp7f3_grouped_only.sh --epochs 3      # quick test
#   ./scripts/run_v7_exp7f3_grouped_only.sh --resume

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
CKPT_DIR="checkpoints_v7_exp7f3_grouped_only"
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"

ARGS="--preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --activation swish --chunk_size $CHUNK_SIZE --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --no_grad_ckpt --no_reverse_assoc"

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
    LOG_DIR=$(make_log_dir "v7" "exp7f3_grouped_only_${DATASET}")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="V7 Exp7f-3 (Grouped hierarchy only): dim=384 L=16 expand=3, PAM(H=6, d=64, grouped dt, RoPE, fused-QKV, GSP, cross_level=True), activation=swish, chunk_size=$CHUNK_SIZE, rev_assoc=False, multi_scale=False, LR=1e-4, warmup=1000"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  V7 Experiment 7f-3: Grouped Hierarchy Only"
echo "  Ablation: grouped dt_bias + cross_level ONLY (vs 7a flat baseline)"
echo "  Preset: $PRESET, ~100M params, B=$BATCH_SIZE"
echo "  dt_bias: grouped (-6.91x4, -5.52x3, -4.08x3, -2.64x3, -1.39x2, 0.0x1)"
echo "  cross_level: True"
echo "  Multi-scale loss: DISABLED"
echo "  Reverse association: DISABLED"
echo "  seq_len: $SEQ_LEN  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Baseline: 7a val PPL 29.73 (flat, uniform dt)"
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
echo "  Experiment 7f-3 complete"
echo "  Wall time: ${elapsed}s ($(printf '%.1f' "$(echo "$elapsed/3600" | bc -l)")h)"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Baseline: 7a val PPL 29.73 (flat, uniform dt)"
echo "============================================================"
