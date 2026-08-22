#!/usr/bin/env bash
# Second half of the 50M scale: matched Transformer baseline (same 75% recall +
# 25% reasoning mix / 50M tokens), then memory + reasoning behavioral probes and
# the gate probe. (Mamba baseline skipped per user decision.)
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
LOG_DIR="logs/v13"
CKPT_DIR="checkpoints_v13/e2b_50m"
TX_CKPT="checkpoints_v13/transformer_50m"
TOKEN_BUDGET="${TOKEN_BUDGET:-50000000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"

CKPT="$CKPT_DIR/best_model.pt"
if [[ ! -f "$CKPT" ]]; then
  CKPT="$CKPT_DIR/final_model.pt"
fi
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: no V13 50M checkpoint in $CKPT_DIR — aborting baselines." >&2
  exit 1
fi

echo "=== Matched Transformer 50M ==="
"$PYTHON" -u scripts/train_matched_baseline.py \
  --arch transformer \
  --size 10m \
  --token_budget "$TOKEN_BUDGET" \
  --batch_size "$BATCH_SIZE" \
  --seq_len "$SEQ_LEN" \
  --lr 3e-4 \
  --pretrain_sources recall,reason \
  --pretrain_weights 3,1 \
  --checkpoint_dir "$TX_CKPT" \
  --save_every_steps 2000 \
  >> "${LOG_DIR}/transformer_50m.log" 2>&1

MEMORY_ARGS=(
  --context-lengths 128,512,1024,2048
  --positions 0,0.5,1
  --association-counts 1,4,8
  --trials 60
  --candidate-count 8
)

echo "=== Memory behavioral: V13-E2b 50M ==="
"$PYTHON" -u scripts/run_memory_behavioral.py \
  --model-type v13 \
  --checkpoint "$CKPT" \
  --preset v13_micro_10m_recall \
  --output "${CKPT_DIR}/behavioral.json" \
  "${MEMORY_ARGS[@]}" \
  >> "${LOG_DIR}/memory_e2b_50m.log" 2>&1

echo "=== Memory behavioral: Transformer 50M ==="
"$PYTHON" -u scripts/run_memory_behavioral.py \
  --model-type transformer \
  --checkpoint "$TX_CKPT/final_model.pt" \
  --output "${TX_CKPT}/behavioral.json" \
  "${MEMORY_ARGS[@]}" \
  >> "${LOG_DIR}/memory_transformer_50m.log" 2>&1

echo "=== Reasoning probe: V13-E2b 50M ==="
"$PYTHON" -u scripts/run_reasoning_behavioral.py \
  --model-type v13 \
  --checkpoint "$CKPT" \
  --preset v13_micro_10m_recall \
  --gaps 0,2,5 \
  --tasks-per-cell 20 \
  --output "${CKPT_DIR}/reasoning.json" \
  >> "${LOG_DIR}/reason_e2b_50m.log" 2>&1

echo "=== Reasoning probe: Transformer 50M ==="
"$PYTHON" -u scripts/run_reasoning_behavioral.py \
  --model-type transformer \
  --checkpoint "$TX_CKPT/final_model.pt" \
  --gaps 0,2,5 \
  --tasks-per-cell 20 \
  --output "${TX_CKPT}/reasoning.json" \
  >> "${LOG_DIR}/reason_transformer_50m.log" 2>&1

echo "=== Gate probe: V13-E2b 50M ==="
"$PYTHON" -u scripts/v13_probe_gates.py \
  --checkpoint "$CKPT" \
  --preset v13_micro_10m_recall \
  --tokens 4096 \
  --out "${CKPT_DIR}/gate_probe.json" \
  >> "${LOG_DIR}/gate_probe_e2b_50m.log" 2>&1

echo "50M scale complete. Compare ${CKPT_DIR} vs ${TX_CKPT} (behavioral + reasoning JSONs)"
