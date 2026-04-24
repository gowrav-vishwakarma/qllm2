#!/usr/bin/env bash
# V8 end-to-end TinyStories smoke for the single-run plan v2.
#
# Trains the ``e2e_tiny_reasoning`` preset from random init, end-to-end, with
# QLC enabled (unsharp_target=True so gamma is non-trivial) and the runtime-
# safe QLC schedule on. This is the sanity gate before launching the medium
# WikiText-103 run; we want to see:
#
#   * loss going down,
#   * mean_gamma > 0 reported in the QLC diagnostic line,
#   * halt(yes/no/cont) not collapsed to a one-hot,
#   * mean_iter > 1 in the late-training phase (t_max ramps to 4).
#
# No --backbone_ckpt: this is a single-run, no Stage A handoff.
#
# Usage:
#   ./scripts/run_v8_e2e_smoke.sh                 # default 3 epochs
#   EPOCHS=5 ./scripts/run_v8_e2e_smoke.sh        # longer smoke
#   ./scripts/run_v8_e2e_smoke.sh --no_schedule   # disable schedule for compare

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v8/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

PRESET="e2e_tiny_reasoning"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"
SCHEDULE="--qlc_schedule"

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no_schedule) SCHEDULE=""; shift ;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

CKPT_DIR="v8/checkpoints/${PRESET}"
LOG_DIR=$(make_log_dir "v8" "${PRESET}_tinystories")
mkdir -p "$CKPT_DIR"

ARGS="--preset $PRESET --dataset tinystories --epochs $EPOCHS --batch_size $BATCH_SIZE --max_samples $MAX_SAMPLES --num_workers 2 --amp_dtype auto --gen_every 0 --diag_every 50 --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR $SCHEDULE"

write_run_info "$LOG_DIR" "V8 e2e single-run smoke ($PRESET) on TinyStories" "$ARGS $EXTRA_ARGS"
echo "============================================================"
echo "  V8 e2e single-run smoke -- $PRESET"
echo "  Epochs: $EPOCHS   Batch: $BATCH_SIZE   Max samples: $MAX_SAMPLES"
echo "  Schedule: ${SCHEDULE:-disabled}"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "============================================================"
eval "$PYTHON_BIN -m v8.train" $ARGS $EXTRA_ARGS

echo
echo "============================================================"
echo "  Smoke complete. Watch for:"
echo "    * mean_gamma > 0 in the QLC diag lines (unsharp_target works)"
echo "    * halt(yes/no/cont) not collapsed to a one-hot"
echo "    * mean_iter > 1 once t_max ramps to 4"
echo "  Gate to medium: gamma > 0 sustained AND PPL not catastrophic."
echo "============================================================"
