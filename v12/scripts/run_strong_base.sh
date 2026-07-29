#!/usr/bin/env bash
# Phase 3: one strong ~100M v12 base (V11 recipe + E3 K=3 + gate-surprisal + chat vocab).
#
# Usage:
#   tmux new -d -s v12_base './v12/scripts/run_strong_base.sh'
#   TOKEN_BUDGET=2000000000 v12/scripts/run_strong_base.sh
#
# After base completes, grow modules with train_curriculum.sh (FREEZE_SHARED=1,
# module_adapter_rank=32) and re-run v12/scripts/permute_eval.sh.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

PY="${PY:-.venv/bin/python}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1000000000}"
BATCH="${BATCH:-8}"
SEQ="${SEQ:-2048}"
CKPT="${CKPT:-checkpoints_v12_strong_base}"
LOG="${LOG:-logs/v12_strong_base/pretrain.log}"

mkdir -p "$(dirname "$LOG")" "$CKPT"

PYTHONUNBUFFERED=1 "$PY" -m v12.train \
  --preset v12_e3_k3_recall --stage pretrain --dataset pretrain_mix \
  --pretrain_sources dclm,fineweb,smoltalk2_mid --pretrain_weights 70,25,5 \
  --blend_warmup_tokens 300000000 \
  --batch_size "$BATCH" --seq_len "$SEQ" --lr 3e-4 --weight_decay 0.01 \
  --token_budget "$TOKEN_BUDGET" --fused_ce \
  --checkpoint_dir "$CKPT" --save_every_steps 5000 \
  2>&1 | tee "$LOG"

"$PY" -m v12.compact --checkpoint "$CKPT/best_model.pt" --out "$CKPT/slim.pt"
"$PY" -m v12.publish --checkpoint "$CKPT/slim.pt" \
  --module_id grammar --version 2.0 --role base \
  --provenance strong_base --registry v12_registry --overwrite

echo "Strong base published as grammar@2.0 — next: FACT with SUBSTRATE=grammar@2.0 FREEZE_SHARED=1 MODULE_ADAPTER_RANK=32"
