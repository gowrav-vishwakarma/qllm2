#!/usr/bin/env bash
# V6 Medium-PAM Experiment (~85M params)
#
# Replaces the vector-state ComplexSSM with Phase-Associative Memory (PAM).
# The state is upgraded to a Complex Matrix (H x d x d), allowing true
# O(d^2) associative capacity per head without interference.
#
# Architecture: Single CGU(expand=3) + PAM(H=6, d=64) + GSP
# Model Dim: 384 (increased from 192 to match param budget)
# Layers: 16
#
# Baseline: medium-rebalanced-gsp val PPL 41.67 (10 epochs).
#
# Usage:
#   ./scripts/run_v6_medium_pam.sh                    # full run
#   ./scripts/run_v6_medium_pam.sh --epochs 3         # quick test
#   ./scripts/run_v6_medium_pam.sh --batch_size 2     # if OOM
#   ./scripts/run_v6_medium_pam.sh --resume            # resume from checkpoint

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=10
SEQ_LEN=2048
DATASET="wikitext103"
SIZE="medium-pam"
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

GEN_PROMPT="In 1923 , the University of"
LOG_DIR=$(make_log_dir "v6" "medium_pam_gsp_${DATASET}")
CKPT_DIR="checkpoints_v6_medium_pam"

ARGS="--dataset $DATASET --size $SIZE --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --no_working_memory --no_internal_memory"

RESUME_ARG=""
if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

echo ""
echo "============================================================"
echo "  V6 Medium-PAM Experiment"
echo "  Architecture: Single CGU(expand=3) + PAM(H=6, d=64) + GSP"
echo "  dim=384  layers=16"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Compare to: medium-rebalanced-gsp (63.2M, Val PPL 41.67)"
echo "============================================================"
echo ""

write_run_info "$LOG_DIR" "Medium-PAM + GSP: dim=384 L=16 expand=3, single_bank=True, PAM(H=6, d=64), GSP=True, no memory" "$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

start_time=$(date +%s)

eval "$PYTHON_BIN -m v6.train" \
    $ARGS \
    --gen_prompt "'$GEN_PROMPT'" \
    --log_dir "$LOG_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    $RESUME_ARG \
    $EXTRA_ARGS

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
mins=$(( elapsed / 60 ))
secs=$(( elapsed % 60 ))

echo ""
echo "============================================================"
echo "  Medium-PAM experiment complete!"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo ""
echo "  Baseline comparisons:"
echo "    medium-rebalanced-gsp  (63.2M): Val PPL 41.67"
echo "    medium-rebalanced      (58.4M): Val PPL 44.47"
echo "    small-rebalanced       (29.5M): Val PPL 52.64"
echo "    GPT-2 124M (Val):               Val PPL ~31"
echo "    GPT-2 124M (Test):              Test PPL ~14.84"
echo "============================================================"
