#!/usr/bin/env bash
# V6 Long-Sequence Experiment
#
# Tests whether longer sequences (2048) improve model quality by
# actually engaging the SSM slow lanes (100K+ token effective span).
#
# Compared against Run 4 baseline (seq_len=512, Val PPL 56.5).
#
# Usage:
#   ./scripts/run_v6_long_seq.sh
#   ./scripts/run_v6_long_seq.sh --epochs 3        # quick test
#   ./scripts/run_v6_long_seq.sh --batch_size 2     # if OOM

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=5
SEQ_LEN=2048
DATASET="wikitext103"
SIZE="small-matched"
BATCH_SIZE=3
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seq_len)    SEQ_LEN="$2";    shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

GEN_PROMPT="In 1923 , the University of"
LOG_DIR=$(make_log_dir "v6" "long_seq_${SEQ_LEN}_${DATASET}")
CKPT_DIR="checkpoints_v6_long_seq/seq${SEQ_LEN}"

echo ""
echo "============================================================"
echo "  V6 Long-Sequence Experiment"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "============================================================"
echo ""

ARGS="--dataset $DATASET --size $SIZE --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode reduce-overhead --amp_dtype auto --num_workers 4 --gen_every 5000 --no_working_memory --no_internal_memory"

write_run_info "$LOG_DIR" "Long-sequence experiment: seq_len=$SEQ_LEN, no memory, next-token objective" "$ARGS --gen_prompt '$GEN_PROMPT' $EXTRA_ARGS"

start_time=$(date +%s)

eval "$PYTHON_BIN -m v6.train" \
    $ARGS \
    --gen_prompt "'$GEN_PROMPT'" \
    --log_dir "$LOG_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    $EXTRA_ARGS

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
mins=$(( elapsed / 60 ))
secs=$(( elapsed % 60 ))

echo ""
echo "============================================================"
echo "  Long-sequence experiment complete!"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo "============================================================"
