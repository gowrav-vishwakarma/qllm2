#!/usr/bin/env bash
# V13 smoke: 10M-param selective PAM on 100% recall curriculum.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="logs/v13"
CKPT_DIR="checkpoints_v13/smoke_recall"
LOG_FILE="${LOG_DIR}/smoke_recall.log"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

TOKEN_BUDGET="${TOKEN_BUDGET:-5000000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEQ_LEN="${SEQ_LEN:-2048}"

echo "V13 smoke recall | budget=${TOKEN_BUDGET} | bs=${BATCH_SIZE}"

PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -u -m v13.train \
  --preset v13_micro_10m_recall \
  --dataset pretrain_mix \
  --pretrain_sources recall \
  --pretrain_weights 1 \
  --token_budget "$TOKEN_BUDGET" \
  --batch_size "$BATCH_SIZE" \
  --seq_len "$SEQ_LEN" \
  --lr 3e-4 \
  --no_grad_ckpt \
  --fused_ce \
  --checkpoint_dir "$CKPT_DIR" \
  --log_dir "$LOG_DIR" \
  --gen_every 0 \
  --save_every_steps 500 \
  >> "$LOG_FILE" 2>&1

CKPT="$CKPT_DIR/best_model.pt"
if [[ ! -f "$CKPT" ]]; then
  CKPT="$CKPT_DIR/final_model.pt"
fi

echo "Running behavioral eval on $CKPT"
"$PYTHON" scripts/run_memory_behavioral.py \
  --model-type v13 \
  --checkpoint "$CKPT" \
  --preset v13_micro_10m_recall \
  --trials 30 \
  --output "${CKPT_DIR}/behavioral.json"

"$PYTHON" scripts/v13_probe_gates.py \
  --checkpoint "$CKPT" \
  --preset v13_micro_10m_recall \
  --tokens 2048 \
  --out "${CKPT_DIR}/gate_probe.json"

"$PYTHON" -m memory_probes --test rank-text \
  --checkpoint "$CKPT" \
  --preset v13_micro_10m_recall \
  --layer 4 \
  --text-tokens 10000 \
  --sample-every 200 \
  --output-dir "${CKPT_DIR}/probes"

echo "Smoke complete. See $LOG_FILE and $CKPT_DIR/"
