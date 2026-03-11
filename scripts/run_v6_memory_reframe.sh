#!/usr/bin/env bash
# V6 Memory Reframe Experiment Suite
#
# Runs 6 sequential experiments to validate the memory reframe architecture.
# Each run gets its own log directory, checkpoint directory, and RUN_INFO.txt.
# Stops on failure; safe to resume by commenting out completed runs.
#
# Usage:
#   ./scripts/run_v6_memory_reframe.sh                    # all 6 runs
#   ./scripts/run_v6_memory_reframe.sh --run_only 1       # run only experiment 1
#   ./scripts/run_v6_memory_reframe.sh --epochs 5         # override epochs for all
#   ./scripts/run_v6_memory_reframe.sh --skip_to 3        # skip runs 1-2, start at run 3
#
# See: EXPERIMENTS_V6_MEMORY_REFRAME.md for full design rationale.

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
EPOCHS=10
SEQ_LEN=512
DATASET="wikitext103"
SIZE="small-matched"
BATCH_SIZE=14
SKIP_TO=1
RUN_ONLY=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seq_len)    SEQ_LEN="$2";    shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --size)       SIZE="$2";       shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --skip_to)    SKIP_TO="$2";    shift 2 ;;
        --run_only)   RUN_ONLY="$2";   shift 2 ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

COMMON="--dataset $DATASET --size $SIZE --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode reduce-overhead --amp_dtype auto --num_workers 4 --gen_every 5000 --no_working_memory --no_internal_memory"

GROUP_DIR=$(make_group_prefix "v6" "memory_reframe_${DATASET}")
echo ""
echo "============================================================"
echo "  V6 Memory Reframe Experiment Suite"
echo "  Group dir: $GROUP_DIR"
echo "  Dataset: $DATASET  Size: $SIZE  Epochs: $EPOCHS"
echo "  Skip to run: $SKIP_TO"
echo "============================================================"
echo ""

run_experiment() {
    local run_num="$1"
    local run_name="$2"
    local description="$3"
    local run_args="$4"

    if [[ $run_num -lt $SKIP_TO ]]; then
        echo "[run $run_num] SKIPPED: $run_name"
        return 0
    fi

    if [[ $RUN_ONLY -gt 0 && $run_num -ne $RUN_ONLY ]]; then
        echo "[run $run_num] SKIPPED (--run_only $RUN_ONLY): $run_name"
        return 0
    fi

    local log_dir="${GROUP_DIR}/run${run_num}_${run_name}"
    local ckpt_dir="checkpoints_v6_reframe/run${run_num}_${run_name}"

    echo ""
    echo "============================================================"
    echo "  Run $run_num / 6: $run_name"
    echo "  $description"
    echo "  Log: $log_dir"
    echo "  Checkpoint: $ckpt_dir"
    echo "============================================================"
    echo ""

    write_run_info "$log_dir" "$description" "$COMMON $run_args $EXTRA_ARGS"

    if [ -d "$ckpt_dir" ] && [ "$(ls -A "$ckpt_dir" 2>/dev/null)" ]; then
        echo "[run $run_num] Clearing old checkpoints in $ckpt_dir/"
        rm -rf "$ckpt_dir"
    fi

    local start_time
    start_time=$(date +%s)

    eval "$PYTHON_BIN -m v6.train" \
        $COMMON \
        $run_args \
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
# Run 1: Control baseline -- no memory, standard next-token
# ---------------------------------------------------------------------------
run_experiment 1 "control_baseline" \
    "Control baseline: no memory, next_token objective, bank_role_weight=0.05" \
    ""

# ---------------------------------------------------------------------------
# Run 2: Span corruption objective -- no memory
# ---------------------------------------------------------------------------
run_experiment 2 "span_corruption" \
    "Span corruption objective (rate=0.15, mean_len=3), no memory" \
    "--objective span_corruption"

# ---------------------------------------------------------------------------
# Run 3: Span corruption + episodic memory (16 slots)
# ---------------------------------------------------------------------------
run_experiment 3 "span_corruption_episodic16" \
    "Span corruption + episodic memory (16 slots)" \
    "--objective span_corruption --episodic_slots 16"

# ---------------------------------------------------------------------------
# Run 4a: Bank role weight = 0.0 (disabled)
# ---------------------------------------------------------------------------
run_experiment 4 "bank_role_0.0" \
    "Bank role weight = 0.0 (diversity loss only, no role pressure)" \
    "--bank_role_weight 0.0"

# ---------------------------------------------------------------------------
# Run 5: Full combination -- span corruption + episodic + bank roles
# ---------------------------------------------------------------------------
run_experiment 5 "full_combination" \
    "Full combination: span_corruption + episodic_slots=16 + bank_role_weight=0.1" \
    "--objective span_corruption --episodic_slots 16 --bank_role_weight 0.1"

# ---------------------------------------------------------------------------
# Run 6: Two-pass model -- bidirectional encoder + causal decoder
# ---------------------------------------------------------------------------
run_experiment 6 "two_pass" \
    "Two-pass model: bidirectional chunk encoder + causal decoder" \
    "--mode two_pass --objective span_corruption"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  All 6 runs complete!"
echo "  Results in: $GROUP_DIR"
echo "============================================================"
echo ""

if [[ -f "${GROUP_DIR}/TIMING.txt" ]]; then
    echo "Timing summary:"
    cat "${GROUP_DIR}/TIMING.txt"
    echo ""
fi

echo "Next steps:"
echo "  1. Compare val PPL across runs"
echo "  2. Check quality metrics (repeat_3gram, restart_frag, unique_word_ratio)"
echo "  3. Read generated samples for qualitative coherence"
echo "  4. Record results in EXPERIMENTS_V6_MEMORY_REFRAME.md section 6"
