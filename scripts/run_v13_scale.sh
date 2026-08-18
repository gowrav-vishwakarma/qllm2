#!/usr/bin/env bash
# V13 100M selective PAM vs matched Transformer on WikiText-103 + recall suite.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
LOG_DIR="logs/v13"
V13_CKPT="checkpoints_v13/scale_100m"
TX_CKPT="checkpoints_v13/transformer_100m"
TOKEN_BUDGET="${TOKEN_BUDGET:-100000000}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p "$LOG_DIR" "$V13_CKPT" "$TX_CKPT"

echo "=== V13 100M scale (${TOKEN_BUDGET} tokens) ==="
"$PYTHON" -u -m v13.train \
  --preset v13_e3_k3_selective \
  --dataset pretrain_mix \
  --pretrain_sources fineweb,recall \
  --pretrain_weights 96,4 \
  --token_budget "$TOKEN_BUDGET" \
  --batch_size "$BATCH_SIZE" \
  --seq_len 2048 \
  --lr 1e-4 \
  --fused_ce \
  --checkpoint_dir "$V13_CKPT" \
  --log_dir "$LOG_DIR" \
  --gen_every 0 \
  --save_every_steps 2000 \
  >> "${LOG_DIR}/scale_v13.log" 2>&1

echo "=== Matched Transformer 100M ==="
"$PYTHON" -u scripts/train_matched_baseline.py \
  --arch transformer \
  --size 100m \
  --token_budget "$TOKEN_BUDGET" \
  --batch_size "$BATCH_SIZE" \
  --seq_len 2048 \
  --lr 1e-4 \
  --pretrain_sources fineweb,recall \
  --pretrain_weights 96,4 \
  --chat_vocab \
  --checkpoint_dir "$TX_CKPT" \
  --save_every_steps 2000 \
  >> "${LOG_DIR}/scale_transformer.log" 2>&1

V13_BEST="$V13_CKPT/best_model.pt"
TX_BEST="$TX_CKPT/final_model.pt"

"$PYTHON" scripts/run_memory_behavioral.py \
  --model-type v13 --checkpoint "$V13_BEST" --preset v13_e3_k3_selective \
  --trials 60 --output "${V13_CKPT}/behavioral.json"

"$PYTHON" scripts/run_memory_behavioral.py \
  --model-type transformer --checkpoint "$TX_BEST" \
  --trials 60 --output "${TX_CKPT}/behavioral.json"

"$PYTHON" scripts/v13_probe_gates.py \
  --checkpoint "$V13_BEST" --preset v13_e3_k3_selective \
  --tokens 4096 --out "${V13_CKPT}/gate_probe.json"

"$PYTHON" -m memory_probes --test rank-text \
  --checkpoint "$V13_BEST" --preset v13_e3_k3_selective \
  --layer 8 --text-tokens 50000 --sample-every 500 \
  --output-dir "${V13_CKPT}/probes"

echo "Scale eval complete. Compare ${V13_CKPT}/behavioral.json vs ${TX_CKPT}/behavioral.json"
