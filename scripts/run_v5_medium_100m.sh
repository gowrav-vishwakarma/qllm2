#!/usr/bin/env bash
# V5 Medium ~100M No-Attention Experiment on WikiText-103
#
# Apples-to-apples comparison against:
#   - Transformer baseline (~100.3M): Val PPL 27.08 (B=3) / 23.13 (B=6)
#   - V6 medium-pam-v3   (~100.4M): Val PPL 29.95
#   - V7 exp 7a           (~100M):  Val PPL 29.73
#
# Uses the EXACT same data pipeline as the transformer baseline:
#   - GPT-2 tokenizer (50257 vocab)
#   - WikiText-103 (wikitext-103-raw-v1)
#   - seq_len = 2048
#   - Same TextDataset chunking
#
# Architecture: V5 AlgebraicLM (ComplexSSM + MultiBank, NO attention)
#   dim=288, state_dim=576, layers=16, banks=2, expand=3
#   ~100.9M params, attn_every_k=0
#
# Training: AdamW, lr=1e-4, warmup=1000, cosine decay, dropout=0.1
#
# Usage:
#   ./scripts/run_v5_medium_100m.sh                    # full run
#   ./scripts/run_v5_medium_100m.sh --epochs 3         # quick test
#   ./scripts/run_v5_medium_100m.sh --batch_size 2     # if OOM
#   ./scripts/run_v5_medium_100m.sh --resume           # resume from checkpoint

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v5/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=10
SEQ_LEN=2048
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

CKPT_DIR="checkpoints_v5_medium_100m"
LOG_DIR=$(make_log_dir "v5" "medium_v5_100m_wikitext103")

ARGS="--dataset wikitext103 --size medium-v5-100m --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --no_attention --init_strategy orthogonal --init_seed 42 --warmup_steps 1000"

RESUME_ARG=""
if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

mkdir -p "$CKPT_DIR"

RUN_DESC="V5 Medium-100M No-Attention: dim=288 state=576 L=16 banks=2 expand=3 (~100.9M), attn=off, LR=1e-4, warmup=1000, WikiText-103"
RUN_ARGS_LINE="$ARGS $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  V5 Medium ~100M No-Attention Experiment"
echo "  Architecture: ComplexSSM + MultiBank (NO attention)"
echo "  dim=288  state=576  layers=16  banks=2  expand=3"
echo "  ~100.9M params, attn_every_k=0"
echo "  Same data pipeline as transformer baseline (WikiText-103)"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo ""
echo "  Compare to:"
echo "    Transformer B=3  (100.3M): Val PPL 27.08"
echo "    Transformer B=6  (100.3M): Val PPL 23.13"
echo "    V6 medium-pam-v3 (100.4M): Val PPL 29.95"
echo "    V7 exp 7a         (~100M): Val PPL 29.73"
echo "============================================================"
echo ""

write_run_info "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"

start_time=$(date +%s)

eval "$PYTHON_BIN -m v5.train" \
    $ARGS \
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
echo "  V5 Medium-100M No-Attention experiment complete!"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo ""
echo "  Baseline comparisons:"
echo "    Transformer B=3  (100.3M): Val PPL 27.08"
echo "    Transformer B=6  (100.3M): Val PPL 23.13"
echo "    V6 medium-pam-v3 (100.4M): Val PPL 29.95"
echo "    V7 exp 7a         (~100M): Val PPL 29.73"
echo "============================================================"
