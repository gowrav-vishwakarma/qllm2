#!/usr/bin/env bash
# V6 Medium-PAM-v3-attn Ablation (~105M params)
#
# Identical to medium-pam-v3 but with PhaseAttention (sliding window, O(n))
# inserted every 4 layers for content-addressable retrieval.
#
# Architecture: [CGU(expand=3) -> PhaseAttn?(every 4) -> PAM(H=6, d=64, RoPE, fused-QKV)] x 16 + GSP
# Attention:    PhaseAttention at layers {3, 7, 11, 15}, heads=6, window=256
# Model Dim:    384
# Layers:       16
# Params:       ~105.1M (+4.7% over baseline 100.4M)
#
# Baseline:
#   medium-pam-v3 (no attention): Val PPL 29.95
#
# Usage:
#   ./scripts/run_v6_medium_pam_v3_attn.sh                    # full run
#   ./scripts/run_v6_medium_pam_v3_attn.sh --epochs 3         # quick test
#   ./scripts/run_v6_medium_pam_v3_attn.sh --batch_size 2     # if OOM
#   ./scripts/run_v6_medium_pam_v3_attn.sh --resume           # resume from checkpoint

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
SIZE="medium-pam-v3-attn"
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
CKPT_DIR="checkpoints_v6_medium_pam_v3_attn"
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"

ARGS="--dataset $DATASET --size $SIZE --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --no_working_memory --no_internal_memory"

RESUME_ARG=""
REUSED_LOG_DIR=0
LOG_DIR=""

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
    LOG_DIR=$(make_log_dir "v6" "medium_pam_v3_attn_${DATASET}")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="Medium-PAM-v3-attn: dim=384 L=16 expand=3, single_bank=True, PAM(H=6, d=64, RoPE, fused-QKV) + PhaseAttn(every=4, heads=6, window=256), GSP=True, interleave_pam=True, LR=1e-4, warmup=1000"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  V6 Medium-PAM-v3-attn Ablation"
echo "  PAM + sparse PhaseAttention (every 4 layers, window=256)"
echo "  Architecture: [CGU(expand=3) -> Attn?(every 4) -> PAM(H=6, d=64)] x16 + GSP"
echo "  dim=384  layers=16  LR=1e-4  warmup=1000"
echo "  Attention: layers {3,7,11,15}  heads=6  window=256"
echo "  Params: ~105.1M (+4.7% over 100.4M baseline)"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Compare to: medium-pam-v3 (100.4M, Val PPL 29.95, no attention)"
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
echo "  Medium-PAM-v3-attn ablation complete!"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo ""
echo "  Baseline comparisons:"
echo "    medium-pam-v3 (no attn)   (100.4M): Val PPL 29.95"
echo "    medium-pam (sequential)   (100.4M): Val PPL 38.95"
echo "    GPT-2 124M (Val):                   Val PPL ~31"
echo "    Mamba-Small 130M:                   Val PPL ~24.1"
echo "============================================================"
