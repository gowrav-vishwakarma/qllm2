#!/usr/bin/env bash
# V6 Medium-Rebalanced + TSO + GSP + HSB Experiment
#
# Builds on medium-rebalanced-gsp (63.2M, val PPL 41.67) by adding
# Holographic State Binding (HSB):
#   - Injects cmul(key, value) bindings directly into SSM state (Bx term)
#   - Retrieves via cmul(query, conj(h)) from SSM state (added to output)
#   - Fully parallel-scan compatible -- no sequential loops, no registers
#   - SSM-native: binding is an intrinsic property of the recurrence
#
# Baseline: medium-rebalanced-gsp val PPL 41.67 (10 epochs).
#
# Usage:
#   ./scripts/run_v6_medium_pcbr.sh                    # full run
#   ./scripts/run_v6_medium_pcbr.sh --epochs 3         # quick test
#   ./scripts/run_v6_medium_pcbr.sh --batch_size 2     # if OOM
#   ./scripts/run_v6_medium_pcbr.sh --resume            # resume from checkpoint

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
SIZE="medium-rebalanced-hsb"
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
LOG_DIR=$(make_log_dir "v6" "medium_hsb_tso_gsp_${DATASET}")
CKPT_DIR="checkpoints_v6_medium_hsb"

ARGS="--dataset $DATASET --size $SIZE --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --no_working_memory --no_internal_memory"

RESUME_ARG=""
if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

echo ""
echo "============================================================"
echo "  V6 Medium-Rebalanced + TSO + GSP + HSB Experiment"
echo "  Architecture: Single CGU(expand=3) + SSM(state_dim=1536)"
echo "                + TSO + GSP + Holographic State Binding"
echo "  dim=192  layers=16  state_dim=1536"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Compare to: medium-rebalanced-gsp (63.2M, Val PPL 41.67)"
echo "============================================================"
echo ""

write_run_info "$LOG_DIR" "Medium-Rebalanced + TSO + GSP + HSB: dim=192 state_dim=1536 L=16 expand=3, single_bank=True, TSO=True, GSP=True, HSB=True, no memory" "$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

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
echo "  Medium-Rebalanced + TSO + GSP + HSB experiment complete!"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo ""
echo "  Baseline comparisons:"
echo "    medium-rebalanced-gsp  (63.2M): Val PPL 41.67"
echo "    medium-rebalanced      (58.4M): Val PPL 44.47"
echo "    small-rebalanced       (29.5M): Val PPL 52.64"
echo "    GPT-2 124M:                     Val PPL ~31"
echo "============================================================"
