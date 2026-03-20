#!/usr/bin/env bash
# V6 Medium-PAM-v3 Experiment (~100M params)
#
# Builds on v2 (interleaved CGU + PAM) with 4 improvements:
#
# QUALITY (ablatable via config flags):
#   1. QK Phase Normalization -- optional; caused repetition when ON (Bug 8,
#      EXPERIMENTS_V6_PART2.md). Preset uses pam_qk_norm=False.
#   2. Complex RoPE on Q,K -- native complex positional encoding via
#      phase rotation e^{i*m*theta}. Fills the zero-position-encoding gap.
#
# SPEED (always on, math-identical):
#   3. Block-Real GEMM -- 1 GEMM per ComplexLinear instead of 4
#   4. Fused QKV -- single projection for Q,K,V in PAM
#
# Architecture: [CGU(expand=3) -> PAM(H=6, d=64, RoPE, fused-QKV)] x 16 + GSP
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
#
# Log directory reuse:
#   On each fresh run, the chosen LOG_DIR is stored in
#   checkpoints_v6_medium_pam_v3/last_log_dir.txt so a later --resume reuses the
#   same folder (append to the same .log). To backfill an old run, create that
#   file with one line: the path to the existing log directory (e.g.
#   logs/v6/medium_pam_v3_rope_wikitext103_20260319_231524_31397f0).

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
CKPT_DIR="checkpoints_v6_medium_pam_v3"
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"

ARGS="--dataset $DATASET --size $SIZE --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --no_working_memory --no_internal_memory"

RESUME_ARG=""
REUSED_LOG_DIR=0
LOG_DIR=""

# Reuse the same log directory on resume when checkpoint exists and sidecar points to a valid dir.
if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" && -f "$LOG_DIR_SIDECAR" ]]; then
    _stored=$(head -n 1 "$LOG_DIR_SIDECAR" | tr -d '\r')
    if [[ -n "$_stored" && -d "$_stored" ]]; then
        LOG_DIR="$_stored"
        REUSED_LOG_DIR=1
        echo "[resume] Reusing log directory from $LOG_DIR_SIDECAR: $LOG_DIR"
    elif [[ -n "$_stored" ]]; then
        echo "[resume] Warning: stored log dir not found on disk: $_stored (will create a new log dir)" >&2
    fi
fi

if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR=$(make_log_dir "v6" "medium_pam_v3_rope_${DATASET}")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="Medium-PAM-v3: dim=384 L=16 expand=3, single_bank=True, PAM(H=6, d=64, pam_qk_norm=False, RoPE, fused-QKV), GSP=True, interleave_pam=True, LR=1e-4, warmup=1000"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  V6 Medium-PAM-v3 Experiment"
echo "  Complex RoPE + Block-Real GEMM + Fused QKV (QK norm OFF in preset)"
echo "  Architecture: [CGU(expand=3) -> PAM(H=6, d=64)] x16 + GSP"
echo "  dim=384  layers=16  LR=1e-4  warmup=1000"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Compare to: medium-pam (100.4M, Val PPL 38.95, sequential)"
echo "============================================================"
echo ""

if [[ $REUSED_LOG_DIR -eq 1 ]]; then
    append_run_info_resume "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"
else
    write_run_info "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"
fi

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
