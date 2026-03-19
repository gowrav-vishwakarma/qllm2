#!/usr/bin/env bash
# V6 Medium-PAM-v3 Experiment (~100M params)
#
# Builds on v2 (interleaved CGU + PAM) with 4 improvements:
#
# QUALITY (ablatable via config flags):
#   1. QK Phase Normalization -- cnormalize Q,K to unit magnitude so
#      attention is purely phase-interference (cosine of phase diffs)
#   2. Complex RoPE on Q,K -- native complex positional encoding via
#      phase rotation e^{i*m*theta}. Fills the zero-position-encoding gap.
#
# SPEED (always on, math-identical):
#   3. Block-Real GEMM -- 1 GEMM per ComplexLinear instead of 4
#   4. Fused QKV -- single projection for Q,K,V in PAM
#
# Architecture: [CGU(expand=3) -> PAM(H=6, d=64, QK-norm, RoPE, fused-QKV)] x 16 + GSP
# Model Dim: 384
# Layers: 16
# Params: ~100.4M (same budget as v1/v2)
#
# Baselines:
#   medium-pam     (sequential, no RoPE, no QK norm): Val PPL 38.95
#   medium-pam-v2  (interleaved, stopped early):      N/A
#
# Usage:
#   ./scripts/run_v6_medium_pam_v3.sh                    # full run
#   ./scripts/run_v6_medium_pam_v3.sh --epochs 3         # quick test
#   ./scripts/run_v6_medium_pam_v3.sh --batch_size 2     # if OOM
#   ./scripts/run_v6_medium_pam_v3.sh --resume            # resume from checkpoint

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
SIZE="medium-pam-v3"
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
LOG_DIR=$(make_log_dir "v6" "medium_pam_v3_qknorm_rope_${DATASET}")
CKPT_DIR="checkpoints_v6_medium_pam_v3"

ARGS="--dataset $DATASET --size $SIZE --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --no_working_memory --no_internal_memory"

RESUME_ARG=""
if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

echo ""
echo "============================================================"
echo "  V6 Medium-PAM-v3 Experiment"
echo "  QK Phase Norm + Complex RoPE + Block-Real GEMM + Fused QKV"
echo "  Architecture: [CGU(expand=3) -> PAM(H=6, d=64)] x16 + GSP"
echo "  dim=384  layers=16  LR=1e-4  warmup=1000"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Compare to: medium-pam (100.4M, Val PPL 38.95, sequential)"
echo "============================================================"
echo ""

write_run_info "$LOG_DIR" "Medium-PAM-v3: dim=384 L=16 expand=3, single_bank=True, PAM(H=6, d=64, QK-norm, RoPE, fused-QKV), GSP=True, interleave_pam=True, LR=1e-4, warmup=1000" "$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

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
echo "  Medium-PAM-v3 experiment complete!"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo ""
echo "  Baseline comparisons:"
echo "    medium-pam (sequential)  (100.4M): Val PPL 38.95"
echo "    medium-rebalanced-gsp    (63.2M):  Val PPL 41.67"
echo "    medium-rebalanced        (58.4M):  Val PPL 44.47"
echo "    GPT-2 124M (Val):          Val PPL ~31"
echo "    GPT-2 124M (Test):         Test PPL ~14.84"
echo "    Mamba-Small 130M:                  Val PPL ~24.1"
echo "============================================================"
