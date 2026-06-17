#!/usr/bin/env bash
# Generic V11 experiment runner (new PAM memory dynamics).
#
# Usage:
#   ./scripts/run_v11_exp.sh <preset> [extra v11.train args...]
#
# Examples:
#   ./scripts/run_v11_exp.sh v11_baseline
#   ./scripts/run_v11_exp.sh v11_e1_perchannel
#   ./scripts/run_v11_exp.sh v11_e2_delta --epochs 3        # smoke gate
#   ./scripts/run_v11_exp.sh v11_e3_multistate --batch_size 12
#
# Defaults match the 7d recipe: B=18, chunk 256, ModSwish, --compile, no grad ckpt.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v11/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET="${1:?usage: run_v11_exp.sh <preset> [extra args]}"
shift || true

EPOCHS=10
SEQ_LEN=2048
DATASET="wikitext103"
BATCH_SIZE=18
CHUNK_SIZE=256
NO_COMPILE=0
CKPT_DIR="checkpoints_${PRESET}"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)         EPOCHS="$2";     shift 2 ;;
        --seq_len)        SEQ_LEN="$2";    shift 2 ;;
        --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
        --chunk_size)     CHUNK_SIZE="$2"; shift 2 ;;
        --dataset)        DATASET="$2";    shift 2 ;;
        --checkpoint_dir) CKPT_DIR="$2";   shift 2 ;;
        --no-compile)     NO_COMPILE=1;    shift ;;
        *)                EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

GEN_PROMPT="In 1923 , the University of"
LOG_DIR=$(make_log_dir "v11" "${PRESET}_${DATASET}")
mkdir -p "$CKPT_DIR"

ARGS="--preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --chunk_size $CHUNK_SIZE --max_samples 9999999 --amp_dtype auto --num_workers 4 --gen_every 5000 --no_grad_ckpt"
if [[ $NO_COMPILE -eq 0 ]]; then
    ARGS="$ARGS --compile --compile_mode default"
fi

RUN_DESC="V11 $PRESET (B=$BATCH_SIZE, chunk=$CHUNK_SIZE): new PAM memory dynamics; control = 7d 26.88"
write_run_info "$LOG_DIR" "$RUN_DESC" "$ARGS --gen_prompt '$GEN_PROMPT' $EXTRA_ARGS"

echo "============================================================"
echo "  V11 experiment: $PRESET"
echo "  B=$BATCH_SIZE chunk=$CHUNK_SIZE epochs=$EPOCHS dataset=$DATASET"
echo "  Log: $LOG_DIR   Ckpt: $CKPT_DIR"
echo "============================================================"

eval "$PYTHON_BIN -m v11.train" \
    $ARGS \
    --gen_prompt "'$GEN_PROMPT'" \
    --log_dir "$LOG_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    $EXTRA_ARGS
