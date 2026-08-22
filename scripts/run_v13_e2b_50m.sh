#!/usr/bin/env bash
# V13-E2b 50M scale: 10x the 5M smoke tokens, 75% recall + 25% reasoning mix.
# Baselines + evals live in run_v13_e2b_50m_rest.sh (run after training).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
LOG_DIR="logs/v13"
CKPT_DIR="checkpoints_v13/e2b_50m"
TOKEN_BUDGET="${TOKEN_BUDGET:-50000000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "=== V13-E2b 50M recall+reason (${TOKEN_BUDGET} tokens, bs=${BATCH_SIZE}, seq=${SEQ_LEN}) ==="
"$PYTHON" -u -m v13.train \
  --preset v13_micro_10m_recall \
  --dataset pretrain_mix \
  --pretrain_sources recall,reason \
  --pretrain_weights 3,1 \
  --token_budget "$TOKEN_BUDGET" \
  --batch_size "$BATCH_SIZE" \
  --seq_len "$SEQ_LEN" \
  --lr 3e-4 \
  --no_grad_ckpt \
  --fused_ce \
  --checkpoint_dir "$CKPT_DIR" \
  --log_dir "$LOG_DIR" \
  --gen_every 0 \
  --save_every_steps 1000 \
  >> "${LOG_DIR}/e2b_50m.log" 2>&1

CKPT="$CKPT_DIR/best_model.pt"
if [[ ! -f "$CKPT" ]]; then
  CKPT="$CKPT_DIR/final_model.pt"
fi
echo "V13 50M training done -> $CKPT (run scripts/run_v13_e2b_50m_rest.sh next)"
