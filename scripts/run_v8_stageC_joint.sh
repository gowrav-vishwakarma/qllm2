#!/usr/bin/env bash
# V8 Stage C — joint fine-tune with KL anchor.
#
# Unfreezes the backbone at low LR (1e-5) with a KL term against the Stage A
# logits to prevent grammar drift while QLC and backbone co-adapt.
# Run this on the top-2 Stage B configs identified by their Val PPL.
#
# Usage:
#   ./scripts/run_v8_stageC_joint.sh stageC_T4_joint     # default top-1 candidate
#   ./scripts/run_v8_stageC_joint.sh <preset> --epochs 3

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v8/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET="${1:-stageC_T4_joint}"
shift || true
BACKBONE_CKPT="${BACKBONE_CKPT:-v8/checkpoints/qpam_stageA.pt}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-3}"
LR="${LR:-1e-5}"
KL="${KL:-0.05}"

if [[ ! -f "$BACKBONE_CKPT" ]]; then
    echo "ERROR: missing backbone checkpoint at $BACKBONE_CKPT"
    exit 1
fi

CKPT_DIR="v8/checkpoints/${PRESET}"
LOG_DIR=$(make_log_dir "v8" "${PRESET}_joint_wikitext103")
mkdir -p "$CKPT_DIR"

ARGS="--preset $PRESET --dataset wikitext103 --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --kl_anchor_weight $KL --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --diag_every 100 --checkpoint_dir $CKPT_DIR --log_dir $LOG_DIR --backbone_ckpt $BACKBONE_CKPT"

write_run_info "$LOG_DIR" "V8 Stage C joint fine-tune ($PRESET)" "$ARGS $*"
echo "============================================================"
echo "  V8 Stage C -- joint fine-tune ($PRESET)"
echo "  Backbone: $BACKBONE_CKPT"
echo "  LR: $LR   KL anchor: $KL   Epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "============================================================"
eval "$PYTHON_BIN -m v8.train" $ARGS "$@"
