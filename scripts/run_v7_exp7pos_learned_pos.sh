#!/usr/bin/env bash
# V7 Experiment 7pos: Learned positional embeddings at input (3-way ablation)
#
# Control (7d B=18 RoPE-only): val PPL 26.88 — already logged, not re-run here.
#
# Variants:
#   hybrid   — learned pos at input + RoPE on PAM Q/K
#   pos_only — learned pos at input, --no_rope
#   both     — run hybrid then pos_only sequentially (default)
#
# Usage:
#   ./scripts/run_v7_exp7pos_learned_pos.sh
#   ./scripts/run_v7_exp7pos_learned_pos.sh --variant hybrid
#   ./scripts/run_v7_exp7pos_learned_pos.sh --variant pos_only
#   ./scripts/run_v7_exp7pos_learned_pos.sh --epochs 3   # quick smoke

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v7/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=10
SEQ_LEN=2048
DATASET="wikitext103"
PRESET="medium_h16_flat"
BATCH_SIZE=18
CHUNK_SIZE=256
ACTIVATION="swish"
VARIANT="both"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seq_len)    SEQ_LEN="$2";    shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --variant)    VARIANT="$2";    shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

BASE_ARGS="--preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --activation $ACTIVATION --chunk_size $CHUNK_SIZE --max_samples 9999999 --amp_dtype auto --num_workers 4 --gen_every 5000 --no_grad_ckpt --compile --compile_mode default --learned_pos"
GEN_PROMPT="In 1923 , the University of"

run_variant() {
    local name="$1"
    local ckpt_dir="$2"
    local run_slug="$3"
    local pos_flags="$4"
    local run_desc="$5"

    local log_dir
    log_dir=$(make_log_dir "v7" "$run_slug")
    mkdir -p "$ckpt_dir"
    printf '%s\n' "$log_dir" > "${ckpt_dir}/last_log_dir.txt"

    local run_args_line="$BASE_ARGS $pos_flags --gen_prompt '$GEN_PROMPT' $EXTRA_ARGS"

    echo ""
    echo "============================================================"
    echo "  V7 Experiment 7pos ($name)"
    echo "  $run_desc"
    echo "  Log: $log_dir"
    echo "  Checkpoint: $ckpt_dir"
    echo "============================================================"
    echo ""

    write_run_info "$log_dir" "$run_desc" "$run_args_line"

    local start_time end_time elapsed
    start_time=$(date +%s)

    eval "$PYTHON_BIN -m v7.train" \
        $BASE_ARGS \
        $pos_flags \
        --gen_prompt "'$GEN_PROMPT'" \
        --log_dir "$log_dir" \
        --checkpoint_dir "$ckpt_dir" \
        $EXTRA_ARGS

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo ""
    echo "============================================================"
    echo "  7pos ($name) complete"
    echo "  Wall time: ${elapsed}s ($(printf '%.1f' "$(echo "$elapsed/3600" | bc -l)")h)"
    echo "  Log: $log_dir"
    echo "============================================================"
}

case "$VARIANT" in
    hybrid)
        run_variant "hybrid" \
            "checkpoints_v7_exp7pos_hybrid" \
            "exp7pos_hybrid_${DATASET}" \
            "" \
            "V7 Exp7pos hybrid: learned pos at input + RoPE on Q/K, dim=384 L=16, ModSwish, chunk=256, B=18, LR=1e-4"
        ;;
    pos_only)
        run_variant "pos_only" \
            "checkpoints_v7_exp7pos_only" \
            "exp7pos_only_${DATASET}" \
            "--no_rope" \
            "V7 Exp7pos pos-only: learned pos at input, RoPE DISABLED, dim=384 L=16, ModSwish, chunk=256, B=18, LR=1e-4"
        ;;
    both)
        run_variant "hybrid" \
            "checkpoints_v7_exp7pos_hybrid" \
            "exp7pos_hybrid_${DATASET}" \
            "" \
            "V7 Exp7pos hybrid: learned pos at input + RoPE on Q/K, dim=384 L=16, ModSwish, chunk=256, B=18, LR=1e-4"
        run_variant "pos_only" \
            "checkpoints_v7_exp7pos_only" \
            "exp7pos_only_${DATASET}" \
            "--no_rope" \
            "V7 Exp7pos pos-only: learned pos at input, RoPE DISABLED, dim=384 L=16, ModSwish, chunk=256, B=18, LR=1e-4"
        ;;
    *)
        echo "Unknown variant: $VARIANT (use hybrid, pos_only, or both)" >&2
        exit 1
        ;;
esac
