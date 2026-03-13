#!/usr/bin/env bash
# V6 Composite Key Memory Experiment
#
# Tests the redesigned episodic memory with composite keys (bilinear bank
# interaction) at seq_len=2048 with large delayed_recall gap.
#
# Runs two experiments:
#   1. No-memory baseline at seq_len=2048 (continue from existing checkpoint)
#   2. Composite key episodic memory + delayed_recall gap=512 at seq_len=2048
#
# Usage:
#   ./scripts/run_v6_composite_keys.sh                    # both runs
#   ./scripts/run_v6_composite_keys.sh --run_only 1       # baseline only
#   ./scripts/run_v6_composite_keys.sh --run_only 2       # composite keys only
#   ./scripts/run_v6_composite_keys.sh --epochs 15        # override epochs
#   ./scripts/run_v6_composite_keys.sh --run_only 2 --resume  # resume run 2 from checkpoint
#   ./scripts/run_v6_composite_keys.sh --run_only 2 --resume --batch_size 6
#
# See: .cursor/plans/v6_memory_deep_analysis_1b91a4d6.plan.md (Steps C & D)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Defaults (override with CLI args)
# ---------------------------------------------------------------------------
EPOCHS=15
SEQ_LEN=2048
DATASET="wikitext103"
SIZE="small-matched"
BATCH_SIZE=6
RUN_ONLY=0
RESUME=0
LOG_DIR_OVERRIDE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seq_len)    SEQ_LEN="$2";    shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --size)       SIZE="$2";       shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --run_only)   RUN_ONLY="$2";   shift 2 ;;
        --resume)     RESUME=1;        shift ;;
        --log_dir)    LOG_DIR_OVERRIDE="$2"; shift 2 ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

GEN_PROMPT="In 1923 , the University of"

COMMON="--dataset $DATASET --size $SIZE --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000"

if [[ $RESUME -eq 1 && -n "$LOG_DIR_OVERRIDE" ]]; then
    # Reuse the provided log dir parent as group dir
    GROUP_DIR="$(dirname "$LOG_DIR_OVERRIDE")"
elif [[ $RESUME -eq 1 ]]; then
    # Auto-find most recent composite_keys group dir
    LATEST_GROUP=$(ls -dt logs/v6/composite_keys_${DATASET}_* 2>/dev/null | head -1)
    if [[ -n "$LATEST_GROUP" ]]; then
        GROUP_DIR="$LATEST_GROUP"
        echo "[resume] Reusing existing group dir: $GROUP_DIR"
    else
        GROUP_DIR=$(make_group_prefix "v6" "composite_keys_${DATASET}")
        echo "[resume] No previous group dir found, creating new: $GROUP_DIR"
    fi
else
    GROUP_DIR=$(make_group_prefix "v6" "composite_keys_${DATASET}")
fi

echo ""
echo "============================================================"
echo "  V6 Composite Key Memory Experiment"
echo "  Group dir: $GROUP_DIR"
echo "  Dataset: $DATASET  Size: $SIZE  Epochs: $EPOCHS"
echo "  Seq len: $SEQ_LEN  Batch size: $BATCH_SIZE"
if [[ $RESUME -eq 1 ]]; then
    echo "  Resume: YES"
fi
echo "============================================================"
echo ""

run_experiment() {
    local run_num="$1"
    local run_name="$2"
    local description="$3"
    local run_args="$4"
    local ckpt_dir="$5"

    if [[ $RUN_ONLY -gt 0 && $run_num -ne $RUN_ONLY ]]; then
        echo "[run $run_num] SKIPPED (--run_only $RUN_ONLY): $run_name"
        return 0
    fi

    local log_dir
    if [[ -n "$LOG_DIR_OVERRIDE" ]]; then
        log_dir="$LOG_DIR_OVERRIDE"
    else
        log_dir="${GROUP_DIR}/run${run_num}_${run_name}"
    fi

    # Auto-detect resume checkpoint
    local resume_arg=""
    if [[ $RESUME -eq 1 && -f "$ckpt_dir/best_model.pt" ]]; then
        resume_arg="--resume $ckpt_dir/best_model.pt"
        echo "[run $run_num] Resuming from $ckpt_dir/best_model.pt"
    elif [[ $RESUME -eq 1 ]]; then
        echo "[run $run_num] --resume requested but no checkpoint found in $ckpt_dir/"
    fi

    echo ""
    echo "============================================================"
    echo "  Run $run_num / 2: $run_name"
    echo "  $description"
    echo "  Log: $log_dir"
    echo "  Checkpoint: $ckpt_dir"
    if [[ -n "$resume_arg" ]]; then
        echo "  Resume: YES (from best_model.pt)"
    fi
    echo "============================================================"
    echo ""

    write_run_info "$log_dir" "$description" "$COMMON --gen_prompt '$GEN_PROMPT' $run_args $resume_arg $EXTRA_ARGS"

    if [ -d "$ckpt_dir" ] && [ "$(ls -A "$ckpt_dir" 2>/dev/null)" ]; then
        echo "[run $run_num] Found existing checkpoints in $ckpt_dir/ -- keeping them"
    fi

    local start_time
    start_time=$(date +%s)

    eval "$PYTHON_BIN -m v6.train" \
        $COMMON \
        --gen_prompt "'$GEN_PROMPT'" \
        $run_args \
        $resume_arg \
        --log_dir "$log_dir" \
        --checkpoint_dir "$ckpt_dir" \
        $EXTRA_ARGS

    local end_time
    end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    echo ""
    echo "[run $run_num] $run_name completed in ${mins}m ${secs}s"
    echo "Run $run_num ($run_name): ${mins}m ${secs}s" >> "${GROUP_DIR}/TIMING.txt"
    echo ""
}

# ---------------------------------------------------------------------------
# Run 1: No-memory baseline at seq_len=2048 (continue from epoch 5)
#   Purpose: establish the PPL ceiling without memory
# ---------------------------------------------------------------------------
run_experiment 1 "no_memory_baseline_2048" \
    "No memory baseline at seq_len=2048 (continued from epoch 5, target: find PPL floor)" \
    "--no_working_memory --no_internal_memory" \
    "checkpoints_v6_long_seq/seq2048"

# ---------------------------------------------------------------------------
# Run 2: Composite key episodic memory + delayed_recall gap=512
#   Purpose: test if composite keys (bilinear bank interaction) fix
#   the phase interference problem and give memory an actual advantage
# ---------------------------------------------------------------------------
run_experiment 2 "composite_keys_episodic16" \
    "Composite key episodic memory: episodic_slots=16, delayed_recall gap=512, bank_role_weight=0.1" \
    "--no_working_memory --no_internal_memory --objective delayed_recall --delayed_recall_gap 512 --episodic_slots 16 --bank_role_weight 0.1" \
    "checkpoints_v6_composite_keys"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Both runs complete!"
echo "  Results in: $GROUP_DIR"
echo "============================================================"
echo ""

if [[ -f "${GROUP_DIR}/TIMING.txt" ]]; then
    echo "Timing summary:"
    cat "${GROUP_DIR}/TIMING.txt"
    echo ""
fi

echo "Next steps:"
echo "  1. Compare val PPL: no-memory baseline vs composite keys"
echo "  2. If composite keys beat baseline -> memory provides value -> proceed to Step E (two-pass)"
echo "  3. If not -> consider structural bank asymmetry or SSM-only scaling path"
