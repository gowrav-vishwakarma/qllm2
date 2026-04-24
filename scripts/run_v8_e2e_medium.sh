#!/usr/bin/env bash
# V8 end-to-end medium WikiText-103 single-run training.
#
# Trains the ``e2e_medium_reasoning`` preset from random init in ONE
# continuous run, no Stage A/B/C handoff. The QLC is enabled the whole way
# with ``unsharp_target=True`` so gamma carries a real signal, and the
# runtime-safe schedule ramps t_max (2 -> 3 -> 4) and ponder_lambda
# (0 -> 0.005 -> 0.01) as training progresses.
#
# Outputs ONE final checkpoint under v8/checkpoints/e2e_medium_reasoning/.
# Hardware: RTX 4090. Wall clock: ~14h for 10 epochs.
#
# Usage:
#   tmux new -s v8_e2e
#   ./scripts/run_v8_e2e_medium.sh                  # full 10-epoch run
#   ./scripts/run_v8_e2e_medium.sh --epochs 1       # smoke
#   EPOCHS=10 BATCH_SIZE=3 ./scripts/run_v8_e2e_medium.sh
#   ./scripts/run_v8_e2e_medium.sh --resume         # resume best checkpoint
#   ./scripts/run_v8_e2e_medium.sh --no_schedule    # use preset values fixed

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v8/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET="e2e_medium_reasoning"
EPOCHS="${EPOCHS:-10}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-3}"
DATASET="${DATASET:-wikitext103}"
SCHEDULE="--qlc_schedule"
RESUME=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)      EPOCHS="$2";     shift 2 ;;
        --seq_len)     SEQ_LEN="$2";    shift 2 ;;
        --batch_size)  BATCH_SIZE="$2"; shift 2 ;;
        --resume)      RESUME=1;        shift ;;
        --no_schedule) SCHEDULE="";     shift ;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

CKPT_DIR="v8/checkpoints/${PRESET}"
LOG_DIR=$(make_log_dir "v8" "${PRESET}_${DATASET}")
mkdir -p "$CKPT_DIR"

GEN_PROMPT="In 1923 , the University of"
ARGS="--preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode reduce-overhead --amp_dtype auto --num_workers 4 --gen_every 5000 --gen_prompt '$GEN_PROMPT' --diag_every 100 --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR $SCHEDULE"

RESUME_ARG=""
if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="V8 e2e single-run on $DATASET (no Stage A handoff, unsharp_target=True, QLC schedule=${SCHEDULE:+on})"
write_run_info "$LOG_DIR" "$RUN_DESC" "$ARGS $RESUME_ARG $EXTRA_ARGS"

echo
echo "============================================================"
echo "  V8 single-run e2e: $PRESET"
echo "  Dataset: $DATASET   seq_len=$SEQ_LEN   batch=$BATCH_SIZE   epochs=$EPOCHS"
echo "  Schedule: ${SCHEDULE:-disabled (preset values fixed)}"
echo "  Log dir: $LOG_DIR"
echo "  Checkpoint dir: $CKPT_DIR"
echo "============================================================"

start=$(date +%s)
eval "$PYTHON_BIN -m v8.train" $ARGS $RESUME_ARG $EXTRA_ARGS
elapsed=$(( $(date +%s) - start ))

echo
echo "============================================================"
echo "  V8 e2e run complete in $((elapsed/60))m $((elapsed%60))s"
echo "  Final checkpoint: $CKPT_DIR/final_model.pt"
echo "  Best checkpoint:  $CKPT_DIR/best_model.pt"
echo "  Reasoning KPIs are in the QLC diag lines of:"
echo "    $LOG_DIR/v8_${PRESET}_${DATASET}.log"
echo "============================================================"
