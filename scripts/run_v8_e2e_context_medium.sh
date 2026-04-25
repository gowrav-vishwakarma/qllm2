#!/usr/bin/env bash
# V8.3 interleaved context-reasoning medium WikiText-103 run.
#
# This launcher runs the new V8.3 preset:
#   e2e_medium_context_reasoning
#
# It differs from run_v8_e2e_medium.sh, which still runs the older
# post-backbone e2e_medium_reasoning preset. Do not resume old V8.2
# checkpoints into this architecture.
#
# Expected diagnostics in the log:
#   QLC[layer_3]
#   QLC[layer_7]
#   QLC[layer_11]
#
# Usage:
#   ./scripts/run_v8_e2e_context_medium.sh
#   ./scripts/run_v8_e2e_context_medium.sh --epochs 1
#   ./scripts/run_v8_e2e_context_medium.sh --no_schedule
#   EPOCHS=10 BATCH_SIZE=3 ./scripts/run_v8_e2e_context_medium.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v8/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET="e2e_medium_context_reasoning"
EPOCHS="${EPOCHS:-10}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-3}"
DATASET="${DATASET:-wikitext103}"
SCHEDULE="--qlc_schedule"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)      EPOCHS="$2";     shift 2 ;;
        --seq_len)     SEQ_LEN="$2";    shift 2 ;;
        --batch_size)  BATCH_SIZE="$2"; shift 2 ;;
        --no_schedule) SCHEDULE="";     shift ;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

CKPT_DIR="v8/checkpoints/${PRESET}"
LOG_DIR=$(make_log_dir "v8" "${PRESET}_${DATASET}")
mkdir -p "$CKPT_DIR"

GEN_PROMPT="In 1923 , the University of"
ARGS="--preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --gen_prompt '$GEN_PROMPT' --diag_every 100 --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR $SCHEDULE"

RUN_DESC="V8.3 interleaved context reasoning on $DATASET (QLC inside backbone layers, target_alignment_target=0.75, schedule=${SCHEDULE:+on})"
write_run_info "$LOG_DIR" "$RUN_DESC" "$ARGS $EXTRA_ARGS"

echo
echo "============================================================"
echo "  V8.3 interleaved context run: $PRESET"
echo "  Dataset: $DATASET   seq_len=$SEQ_LEN   batch=$BATCH_SIZE   epochs=$EPOCHS"
echo "  Schedule: ${SCHEDULE:-disabled (preset values fixed)}"
echo "  Log dir: $LOG_DIR"
echo "  Checkpoint dir: $CKPT_DIR"
echo "  Expected QLC diagnostics: QLC[layer_3], QLC[layer_7], QLC[layer_11]"
echo "============================================================"

start=$(date +%s)
eval "$PYTHON_BIN -m v8.train" $ARGS $EXTRA_ARGS
elapsed=$(( $(date +%s) - start ))

echo
echo "============================================================"
echo "  V8.3 context run complete in $((elapsed/60))m $((elapsed%60))s"
echo "  Final checkpoint: $CKPT_DIR/final_model.pt"
echo "  Best checkpoint:  $CKPT_DIR/best_model.pt"
echo "  Reasoning KPIs are in the QLC[layer_*] diag lines of:"
echo "    $LOG_DIR/v8_${PRESET}_${DATASET}.log"
echo "============================================================"
