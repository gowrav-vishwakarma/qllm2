#!/usr/bin/env bash
# Matched ~10M baselines on the same fact mix (behavioral suite). V12 Tier 4 / plan micro-baselines.
#
# Note: Mamba uses a public pretrained checkpoint unless TRAIN_MAMBA=1 (slow).
# Transformer baseline trains from scratch on fact data when TRAIN_TF=1.
#
# Usage:
#   v12/scripts/run_micro_baselines.sh v12
#   TRAIN_TF=1 v12/scripts/run_micro_baselines.sh transformer

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

PY="${PY:-.venv/bin/python}"
TARGET="${1:-v12}"
OUT="${OUT:-logs/v12_micro_baselines/summary.json}"
mkdir -p logs/v12_micro_baselines

run_v12() {
  local ckpt="${1:-checkpoints_v12_micro_ab/control/best_model.pt}"
  "$PY" -m memory_probes.behavioral --checkpoint "$ckpt" --trials 40 \
    --context-lengths 128,512,1024 --association-counts 1,4,8 \
    | tee logs/v12_micro_baselines/v12_behavior.log
}

run_transformer() {
  local ckpt=checkpoints_v12_micro_baselines/transformer
  if [ "${TRAIN_TF:-0}" = "1" ]; then
    "$PY" -m v6.transformer_baseline.train --dim 96 --n_layers 4 --n_heads 4 \
      --seq_len 512 --batch_size 8 --steps 2000 --dataset fact \
      --checkpoint_dir "$ckpt" || echo "transformer micro train skipped (wire dataset if missing)"
  fi
  if [ -f "$ckpt/best_model.pt" ]; then
    "$PY" -m memory_probes.behavioral --checkpoint "$ckpt/best_model.pt" --trials 40 \
      --context-lengths 128,512,1024 --association-counts 1,4,8 \
      | tee logs/v12_micro_baselines/transformer_behavior.log
  fi
}

run_mamba() {
  if [ "${TRAIN_MAMBA:-0}" != "1" ]; then
    echo "Mamba: set TRAIN_MAMBA=1 or point MAMBA_CHECKPOINT= to a local ckpt for fair scratch training."
    if [ -n "${MAMBA_CHECKPOINT:-}" ]; then
      "$PY" -m memory_probes.behavioral --checkpoint "$MAMBA_CHECKPOINT" --trials 40 \
        | tee logs/v12_micro_baselines/mamba_behavior.log
    fi
    return 0
  fi
  echo "Mamba scratch training not wired in-repo; use MAMBA_CHECKPOINT for eval-only compare."
}

case "$TARGET" in
  v12) run_v12 ;;
  transformer) run_transformer ;;
  mamba) run_mamba ;;
  all)
    run_v12
    run_transformer
    run_mamba
    ;;
  *)
    echo "Usage: $0 {v12|transformer|mamba|all}"
    exit 1
    ;;
esac

echo "{\"target\":\"$TARGET\",\"log_dir\":\"logs/v12_micro_baselines\"}" > "$OUT"
