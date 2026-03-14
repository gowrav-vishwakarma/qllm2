#!/usr/bin/env bash
# V6 Rebalanced Architecture Experiment (Option B + TSO)
#
# Tests the restructured model: single CGU per layer (no dual banks/coupler)
# with doubled SSM state_dim (1280 vs 512) and Timescale-Separated Output.
#
# Parameter budget stays ~29M but reallocates from banks (32% -> 16%)
# to SSM (16% -> 33%), doubling the model's temporal processing capacity.
#
# Compared against Run 4 baseline (small-matched, 10 epochs, Val PPL 56.46).
#
# Usage:
#   ./scripts/run_v6_rebalanced.sh                    # full run
#   ./scripts/run_v6_rebalanced.sh --epochs 3         # quick test
#   ./scripts/run_v6_rebalanced.sh --batch_size 2     # if OOM
#   ./scripts/run_v6_rebalanced.sh --resume            # resume from checkpoint

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
SIZE="small-rebalanced"
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
LOG_DIR=$(make_log_dir "v6" "rebalanced_tso_${DATASET}")
CKPT_DIR="checkpoints_v6_rebalanced"

ARGS="--dataset $DATASET --size $SIZE --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --no_working_memory --no_internal_memory"

RESUME_ARG=""
if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

echo ""
echo "============================================================"
echo "  V6 Rebalanced + TSO Experiment"
echo "  Architecture: Single CGU + SSM(state_dim=1280) + TSO"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Compare to: Run 4 baseline (small-matched, Val PPL 56.46)"
echo "============================================================"
echo ""

write_run_info "$LOG_DIR" "Rebalanced Option B + TSO: single_bank=True, state_dim=1280, TSO=True, no memory" "$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

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
echo "  Rebalanced + TSO experiment complete!"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo ""
echo "  Baseline comparison (Run 4): Val PPL 56.46"
echo "  If better -> rebalancing works, consider scaling to 60M"
echo "  If same   -> SSM capacity was not the bottleneck"
echo "============================================================"
