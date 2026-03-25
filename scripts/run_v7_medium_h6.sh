#!/usr/bin/env bash
# V7 Hierarchical PAM (medium_h6, ~102M params) on WikiText-103
#
# Novel architecture: 6-layer hierarchical Phase-Associative Memory.
# Each PAM layer specializes in a distinct temporal scope:
#
#   Layer 0  book       dt_bias=-6.91  span ~1000 tok  (article theme)
#   Layer 1  section    dt_bias=-5.52  span  ~250 tok  (multi-paragraph)
#   Layer 2  chapter    dt_bias=-4.08  span   ~60 tok  (full paragraph)
#   Layer 3  paragraph  dt_bias=-2.64  span   ~15 tok  (couple sentences)
#   Layer 4  sentence   dt_bias=-1.39  span    ~5 tok  (local syntax)
#   Layer 5  word       dt_bias= 0.00  span    ~2 tok  (next-token)
#
# Architecture: [CGU(expand=4) -> PAM(H=8, d=64, hierarchical dt)] x 6
# Model Dim: 512, Heads: 8, Head Dim: 64
# Params: ~102.4M
#
# Baselines:
#   medium-pam-v3  (100.4M, 16-layer uniform): Val PPL ~30
#   Transformer    (100.3M, 12-layer):          Val PPL ~31
#   GPT-2 124M (reported):                      Val PPL ~31
#   Mamba-Small 130M:                           Val PPL ~24.1
#
# Usage:
#   ./scripts/run_v7_medium_h6.sh                    # full run
#   ./scripts/run_v7_medium_h6.sh --epochs 3         # quick test
#   ./scripts/run_v7_medium_h6.sh --batch_size 2     # if OOM
#   ./scripts/run_v7_medium_h6.sh --resume           # resume from checkpoint

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
PRESET="medium_h6"
BATCH_SIZE=3
RESUME=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seq_len)    SEQ_LEN="$2";    shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --resume)     RESUME=1;        shift ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

GEN_PROMPT="In 1923 , the University of"
CKPT_DIR="checkpoints_v7_${PRESET}"
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"

ARGS="--preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000"

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
    LOG_DIR=$(make_log_dir "v7" "${PRESET}_${DATASET}")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="V7 Hierarchical PAM (medium_h6): dim=512 L=6 expand=4, PAM(H=8, d=64, hierarchical_dt, RoPE, fused-QKV, GSP), book->word timescale, LR=1e-4, warmup=1000"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  V7 Hierarchical PAM (medium_h6, ~102M params)"
echo "  6 layers: book -> section -> chapter -> paragraph -> sentence -> word"
echo "  Architecture: [CGU(expand=4) -> PAM(H=8, d=64)] x6 + GSP + RoPE"
echo "  dim=512  heads=8  LR=1e-4  warmup=1000"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Dataset: $DATASET"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "============================================================"
echo ""

if [[ $REUSED_LOG_DIR -eq 1 ]]; then
    append_run_info_resume "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"
else
    write_run_info "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"
fi

start_time=$(date +%s)

eval "$PYTHON_BIN -m v7.train" \
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
echo "  V7 Hierarchical PAM experiment complete!"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo ""
echo "  Baseline comparisons:"
echo "    medium-pam-v3    (100.4M, 16L uniform): Val PPL ~30"
echo "    Transformer      (100.3M, 12L):         Val PPL ~31"
echo "    GPT-2 124M (reported):                   Val PPL ~31"
echo "    Mamba-Small 130M:                        Val PPL ~24.1"
echo "============================================================"
