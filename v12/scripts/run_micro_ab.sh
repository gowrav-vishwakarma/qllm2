#!/usr/bin/env bash
# Micro-lab A/B for M3 levers (~30M tokens/arm). Gate before any 50M+ spend.
#
# Usage:
#   ARM=control v12/scripts/run_micro_ab.sh
#   ARM=backsub_vault v12/scripts/run_micro_ab.sh
#
# Env: TOKEN_BUDGET=30000000 BATCH=4 SEQ=512

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

PY="${PY:-.venv/bin/python}"
ARM="${ARM:-control}"
TOKEN_BUDGET="${TOKEN_BUDGET:-30000000}"
BATCH="${BATCH:-4}"
SEQ="${SEQ:-512}"
CKPT="${CKPT:-checkpoints_v12_micro_ab/${ARM}}"
LOG="${LOG:-logs/v12_micro_ab/${ARM}.log}"

mkdir -p "$(dirname "$LOG")" "$CKPT"

case "$ARM" in
  control)
    PRESET=v12_grammar_dyn_micro
    EXTRA=()
    ;;
  backsub_vault)
    PRESET=v12_micro_factband
    EXTRA=(--vault_norm_bound 8.0 --write_phase_key_conditional)
    ;;
  key_phase)
    PRESET=v12_grammar_dyn_micro
    EXTRA=(--write_phase_key_conditional)
    ;;
  *)
    echo "Unknown ARM=$ARM (control|backsub_vault|key_phase)"
    exit 1
    ;;
esac

echo "+ micro A/B arm=$ARM preset=$PRESET -> $CKPT"
PYTHONUNBUFFERED=1 "$PY" -m v12.train \
  --preset "$PRESET" --stage pretrain --dataset fact \
  --stage_loss ce_fact --fused_ce \
  --batch_size "$BATCH" --seq_len "$SEQ" --lr 3e-5 \
  --token_budget "$TOKEN_BUDGET" \
  --checkpoint_dir "$CKPT" --save_every_steps 500 \
  --gen_every 0 "${EXTRA[@]}" 2>&1 | tee "$LOG"

"$PY" -m v12.diagnose_fact_shortcut --grid --trials 60 \
  --checkpoint "$CKPT/best_model.pt" --seq_len "$SEQ" \
  --output "logs/v12_micro_ab/grid_${ARM}.json"

"$PY" -m memory_probes.behavioral --checkpoint "$CKPT/best_model.pt" \
  --trials 20 --context-lengths 128,512 --association-counts 1,8 \
  > "logs/v12_micro_ab/behavior_${ARM}.json" 2>&1 || true
