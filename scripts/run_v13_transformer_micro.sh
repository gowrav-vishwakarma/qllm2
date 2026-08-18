#!/usr/bin/env bash
# Matched Transformer micro (~10M) on 100% recall — same curriculum as V13 smoke.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="logs/v13"
CKPT_DIR="checkpoints_v13/transformer_micro_recall"
LOG_FILE="${LOG_DIR}/transformer_micro_recall.log"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

TOKEN_BUDGET="${TOKEN_BUDGET:-5000000}"
BATCH_SIZE="${BATCH_SIZE:-8}"

PYTHON="${PYTHON:-.venv/bin/python}"

echo "Transformer micro recall | budget=${TOKEN_BUDGET} | bs=${BATCH_SIZE}"

"$PYTHON" -u scripts/train_matched_baseline.py \
  --arch transformer \
  --size 10m \
  --token_budget "$TOKEN_BUDGET" \
  --batch_size "$BATCH_SIZE" \
  --seq_len 512 \
  --lr 3e-4 \
  --warmup_steps 500 \
  --pretrain_sources recall \
  --pretrain_weights 1 \
  --chat_vocab \
  --checkpoint_dir "$CKPT_DIR" \
  --save_every_steps 500 \
  >> "$LOG_FILE" 2>&1

CKPT="$CKPT_DIR/final_model.pt"
"$PYTHON" scripts/run_memory_behavioral.py \
  --model-type transformer \
  --checkpoint "$CKPT" \
  --trials 30 \
  --output "${CKPT_DIR}/behavioral.json"

echo "Transformer micro complete. See $LOG_FILE"
