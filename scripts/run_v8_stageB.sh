#!/usr/bin/env bash
# V8 Stage B — frozen-backbone QLC ablation sweep.
#
# Designed to be launched as a single overnight session on a single A100 80GB:
# the script schedules each row sequentially (or with explicit GPU pinning if
# you have multiple GPUs). Each row uses ~12GB peak VRAM, so 5-6 rows can fit
# in parallel if you wrap calls with `CUDA_VISIBLE_DEVICES`.
#
# By default this script runs ONE row (passed as $1) so you can fan it out
# yourself across multiple shells / GPUs / A100 MIG slices. Without args it
# prints the available rows and exits.
#
# Usage:
#   ./scripts/run_v8_stageB.sh                         # list rows
#   ./scripts/run_v8_stageB.sh stageB_T4               # run one row
#   ./scripts/run_v8_stageB.sh all                     # run all rows sequentially
#
# Required: v8/checkpoints/qpam_stageA.pt (created by run_v8_stageA.sh).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v8/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALL_ROWS=(
    stageB_r4
    stageB_r8
    stageB_r16
    stageB_M2k
    stageB_M8k
    stageB_T2
    stageB_T4
    stageB_F_quantale_off
    stageB_G_orthohalt_off
)

BACKBONE_CKPT="${BACKBONE_CKPT:-v8/checkpoints/qpam_stageA.pt}"
EPOCHS="${EPOCHS:-5}"     # fewer epochs since backbone is frozen
BATCH_SIZE="${BATCH_SIZE:-3}"
SEQ_LEN="${SEQ_LEN:-2048}"
LR="${LR:-3e-4}"          # higher LR is fine for the small QLC param set

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <row|all>"
    echo "Available rows:"
    for r in "${ALL_ROWS[@]}"; do echo "  - $r"; done
    exit 0
fi

if [[ ! -f "$BACKBONE_CKPT" ]]; then
    echo "ERROR: missing backbone checkpoint at $BACKBONE_CKPT"
    echo "Run scripts/run_v8_stageA.sh first."
    exit 1
fi

run_one() {
    local ROW="$1"
    local CKPT_DIR="v8/checkpoints/${ROW}"
    local LOG_DIR
    LOG_DIR=$(make_log_dir "v8" "${ROW}_wikitext103")
    mkdir -p "$CKPT_DIR"

    local ARGS="--preset $ROW --dataset wikitext103 --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 0 --diag_every 100 --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR --backbone_ckpt $BACKBONE_CKPT"

    write_run_info "$LOG_DIR" "V8 Stage B row $ROW (frozen backbone)" "$ARGS"
    echo "============================================================"
    echo "  V8 Stage B -- $ROW"
    echo "  Preset: $ROW   Backbone: $BACKBONE_CKPT"
    echo "  Epochs: $EPOCHS   Batch: $BATCH_SIZE   LR: $LR"
    echo "  Log: $LOG_DIR"
    echo "============================================================"
    eval "$PYTHON_BIN -m v8.train" $ARGS
}

if [[ "$1" == "all" ]]; then
    for ROW in "${ALL_ROWS[@]}"; do
        run_one "$ROW"
    done
else
    if [[ ! " ${ALL_ROWS[*]} " =~ " $1 " ]]; then
        echo "Unknown row: $1"
        echo "Available: ${ALL_ROWS[*]}"
        exit 1
    fi
    run_one "$1"
fi
