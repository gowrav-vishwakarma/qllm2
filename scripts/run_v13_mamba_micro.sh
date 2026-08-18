#!/usr/bin/env bash
# Optional: matched Mamba tiny on 100% recall (reference point, not primary target).
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="logs/v13"
CKPT_DIR="checkpoints_v13/mamba_micro_recall"
LOG_FILE="${LOG_DIR}/mamba_micro_recall.log"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

TOKEN_BUDGET="${TOKEN_BUDGET:-5000000}"
BATCH_SIZE="${BATCH_SIZE:-4}"

PYTHON="${PYTHON:-.venv/bin/python}"

echo "Mamba micro recall | budget=${TOKEN_BUDGET} | bs=${BATCH_SIZE}"

"$PYTHON" -u scripts/train_matched_baseline.py \
  --arch mamba \
  --size tiny \
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

"$PYTHON" scripts/run_memory_behavioral.py \
  --model-type hf \
  --model-id "$CKPT_DIR/best_hf" \
  --trials 30 \
  --output "${CKPT_DIR}/behavioral.json"

echo "Mamba micro complete. See $LOG_FILE"
