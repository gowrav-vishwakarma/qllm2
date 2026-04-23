#!/usr/bin/env bash
# V8 Stage A — train QPAM backbone from scratch on WikiText-103.
#
# This is the apples-to-apples reproduction of `medium-pam-v3` (Val PPL ≈ 29.95)
# living inside the V8 codebase, so the resulting checkpoint is loadable
# directly into `V8LM` via `--backbone_ckpt` for Stage B / C.
#
# Hardware: RTX 4090, ~14h for 10 epochs.
#
# Usage:
#   ./scripts/run_v8_stageA.sh                  # full run
#   ./scripts/run_v8_stageA.sh --epochs 1       # smoke test
#   ./scripts/run_v8_stageA.sh --resume         # resume from checkpoint

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v8/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=10
SEQ_LEN=2048
DATASET="wikitext103"
PRESET="stageA_medium"
BATCH_SIZE=3
RESUME=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seq_len)    SEQ_LEN="$2";    shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --resume)     RESUME=1;        shift ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

CKPT_DIR="v8/checkpoints/stageA"
LOG_DIR=$(make_log_dir "v8" "stageA_${PRESET}_${DATASET}")
mkdir -p "$CKPT_DIR"

GEN_PROMPT="In 1923 , the University of"
ARGS="--preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode reduce-overhead --amp_dtype auto --num_workers 4 --gen_every 5000 --gen_prompt '$GEN_PROMPT' --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR"

RESUME_ARG=""
if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="V8 Stage A: QPAM backbone from scratch (medium-pam-v3-equivalent), target Val PPL ≈ 29.95"
write_run_info "$LOG_DIR" "$RUN_DESC" "$ARGS $RESUME_ARG $EXTRA_ARGS"

echo
echo "============================================================"
echo "  V8 Stage A: QPAM backbone pretrain"
echo "  Preset: $PRESET   Dataset: $DATASET   seq_len=$SEQ_LEN"
echo "  Batch: $BATCH_SIZE   Epochs: $EPOCHS"
echo "  Log dir: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Target: Val PPL ≈ 29.95 ± 0.5  (medium-pam-v3 baseline)"
echo "============================================================"

start=$(date +%s)
eval "$PYTHON_BIN -m v8.train" $ARGS $RESUME_ARG $EXTRA_ARGS
elapsed=$(( $(date +%s) - start ))

# Promote the best checkpoint to the canonical Stage A path used by Stage B / C.
if [[ -f "$CKPT_DIR/best_model.pt" ]]; then
    cp -f "$CKPT_DIR/best_model.pt" "v8/checkpoints/qpam_stageA.pt"
    echo "[promote] Copied best_model.pt -> v8/checkpoints/qpam_stageA.pt"
fi

echo
echo "============================================================"
echo "  V8 Stage A complete in $((elapsed/60))m $((elapsed%60))s"
echo "  Stage B can now load: --backbone_ckpt v8/checkpoints/qpam_stageA.pt"
echo "============================================================"
