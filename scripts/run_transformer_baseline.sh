#!/usr/bin/env bash
# Transformer Baseline (~100M params) on WikiText-103
#
# Standard GPT-2-style decoder-only transformer for apples-to-apples
# comparison against medium-pam-v3 (~100.4M params).
#
# Uses the EXACT same data pipeline:
#   - GPT-2 tokenizer (50257 vocab)
#   - WikiText-103 (wikitext-103-raw-v1)
#   - seq_len = 2048
#   - Same TextDataset chunking and evaluation loop
#
# Architecture: d_model=672, n_layers=12, n_heads=12, d_ff=2688 (~100.3M params)
# Training: AdamW, lr=1e-4, warmup=1000, cosine decay, dropout=0.1
#
# Baselines:
#   medium-pam-v3 (100.4M, interleaved CGU+PAM, RoPE): Val PPL ~30
#   GPT-2 124M (reported):                              Val PPL ~31
#
# Usage:
#   ./scripts/run_transformer_baseline.sh                    # full run
#   ./scripts/run_transformer_baseline.sh --epochs 3         # quick test
#   ./scripts/run_transformer_baseline.sh --batch_size 2     # if OOM
#   ./scripts/run_transformer_baseline.sh --resume           # resume from checkpoint

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
CKPT_DIR="checkpoints_transformer_baseline"
LOG_DIR=$(make_log_dir "v6" "transformer_baseline_wikitext103")

ARGS="--seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000"

RESUME_ARG=""
if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

mkdir -p "$CKPT_DIR"

RUN_DESC="Transformer Baseline: d_model=672 n_layers=12 n_heads=12 d_ff=2688 (~100.3M), LR=1e-4, warmup=1000, WikiText-103"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  Transformer Baseline (~100M params)"
echo "  GPT-2-style: d_model=672  n_layers=12  n_heads=12  d_ff=2688"
echo "  Same data pipeline as PAM v3 (WikiText-103, GPT-2 tokenizer)"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo ""
echo "  Compare to:"
echo "    medium-pam-v3    (100.4M): Val PPL ~30"
echo "    medium-pam       (100.4M): Val PPL 38.95"
echo "    GPT-2 124M (reported):     Val PPL ~31"
echo "============================================================"
echo ""

write_run_info "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"

start_time=$(date +%s)

eval "$PYTHON_BIN -m v6.train_transformer_baseline" \
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
echo "  Transformer baseline experiment complete!"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo ""
echo "  Baseline comparisons:"
echo "    medium-pam-v3    (100.4M): Val PPL ~30"
echo "    medium-pam       (100.4M): Val PPL 38.95"
echo "    GPT-2 124M (reported):     Val PPL ~31"
echo "    Mamba-Small 130M:          Val PPL ~24.1"
echo "============================================================"
