#!/usr/bin/env bash
# V8 Stage A.5 — TinyStories smoke test for the QLC primitive.
#
# Goal: confirm the QLC primitive (SPM + bank + halt + reasoning loop) is
# differentiable, doesn't NaN, and is no worse than passthrough on a tiny
# corpus. This is the GATE before spending A100 hours on Stage B.
#
# Acceptance: passthrough preset val PPL is consistent with the expected
# TinyStories baseline (~5-7 PPL at the tiny config); QLC variant within
# +1 PPL of passthrough.
#
# Usage:
#   ./scripts/run_v8_stageA5_smoke.sh             # both passthrough and QLC
#   ./scripts/run_v8_stageA5_smoke.sh passthrough # just one variant

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v8/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

WHICH="${1:-both}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"

run_one() {
    local PRESET="$1"
    local LABEL="$2"
    local CKPT_DIR="v8/checkpoints/stageA5_${LABEL}"
    local LOG_DIR
    LOG_DIR=$(make_log_dir "v8" "stageA5_${LABEL}_tinystories")
    mkdir -p "$CKPT_DIR"

    local ARGS="--preset $PRESET --dataset tinystories --epochs $EPOCHS --batch_size $BATCH_SIZE --max_samples $MAX_SAMPLES --num_workers 2 --amp_dtype auto --gen_every 0 --diag_every 50 --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR"

    write_run_info "$LOG_DIR" "V8 Stage A.5 smoke ($LABEL) on TinyStories" "$ARGS"
    echo "============================================================"
    echo "  V8 Stage A.5 smoke -- $LABEL"
    echo "  Preset: $PRESET   Epochs: $EPOCHS   Batch: $BATCH_SIZE"
    echo "  Log: $LOG_DIR"
    echo "============================================================"
    eval "$PYTHON_BIN -m v8.train" $ARGS
}

if [[ "$WHICH" == "passthrough" || "$WHICH" == "both" ]]; then
    run_one smoke_tiny_passthrough passthrough
fi
if [[ "$WHICH" == "qlc" || "$WHICH" == "both" ]]; then
    run_one smoke_tiny_qlc_r4_T2 qlc_r4_T2
fi

echo
echo "============================================================"
echo "  Stage A.5 smoke done. Compare val PPLs in the two log dirs."
echo "  Gate to A100: |QLC - passthrough| ≤ +1 PPL  =>  proceed."
echo "============================================================"
