#!/usr/bin/env bash
# Phase 0 (V12 next phase): 50-value control, pool sweep, full-strength contrastive.
# Matches attempt-C settings except FACT_VALUE_POOL and optional ce_fact_strong loss.
#
# Usage:
#   v12/scripts/run_phase0_controls.sh control50     # ~80 min on 4090
#   v12/scripts/run_phase0_controls.sh sweep           # 50, 200, 1000 sequential
#   v12/scripts/run_phase0_controls.sh contrastive     # fact_contrastive_lambda=0.5
#   v12/scripts/run_phase0_controls.sh grid control50  # transfer grid on checkpoint
#
# Env: SUBSTRATE=grammar@1.0 VER=3.0 CKPT_ROOT=checkpoints_v12_phase0

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

PY="${PY:-.venv/bin/python}"
REGISTRY="${REGISTRY:-v12_registry}"
SUBSTRATE="${SUBSTRATE:-grammar@1.0}"
VER="${VER:-3.0}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints_v12_phase0}"
LOG_ROOT="${LOG_ROOT:-logs/v12_phase0}"
GRID_DIR="${LOG_ROOT}/grid"

mkdir -p "$LOG_ROOT" "$GRID_DIR"

_attempt_c_env() {
  export BATCH=8 SEQ=512 TOKEN_BUDGET=40000000 FACT_LR=3e-5
  export GEN_EVERY=0 SAVE_EVERY_STEPS_FACT=1000
  export SUBSTRATE CKPT_ROOT REGISTRY VER
  export DATASET=fact STAGE_LOSS=ce_fact FACT_MODE=delta FACT_LAYERS=4
}

train_fact() {
  local tag="$1" pool="$2" freeze="${3:-0}" loss="${4:-ce_fact}"
  _attempt_c_env
  export FACT_VALUE_POOL="$pool"
  export FREEZE_SHARED="$freeze"
  export STAGE_LOSS="$loss"
  local ckpt="$CKPT_ROOT/${tag}/fact_retrieval"
  local log="$LOG_ROOT/${tag}_pool${pool}.log"
  echo "=== train $tag pool=$pool freeze=$freeze loss=$loss -> $log"
  PYTHONUNBUFFERED=1 \
    CKPT_ROOT="$CKPT_ROOT/$tag" \
    v12/scripts/train_curriculum.sh fact_retrieval 2>&1 | tee "$log"
}

run_grid() {
  local tag="$1" ckpt="$CKPT_ROOT/${tag}/fact_retrieval/best_model.pt"
  local out="$GRID_DIR/${tag}.json"
  echo "=== grid $tag -> $out"
  "$PY" -m v12.diagnose_fact_shortcut --grid --trials 200 \
    --checkpoint "$ckpt" --seq_len 512 --value_pool "${2:-50}" \
    --output "$out"
}

case "${1:-help}" in
  control50)
    train_fact control50 50 0 ce_fact
    run_grid control50 50
    ;;
  control50_frozen)
    train_fact control50_frozen 50 1 ce_fact
    run_grid control50_frozen 50
    ;;
  pool200)
    train_fact pool200 200 0 ce_fact
    run_grid pool200 200
    ;;
  pool1000)
    train_fact pool1000 1000 0 ce_fact
    run_grid pool1000 1000
    ;;
  contrastive)
    train_fact contrastive 50 0 ce_fact_strong
    run_grid contrastive 50
    ;;
  sweep)
    for p in 50 200 1000; do
      train_fact "pool${p}" "$p" 0 ce_fact
      run_grid "pool${p}" "$p"
    done
    ;;
  grid)
    run_grid "${2:?tag}" "${3:-50}"
    ;;
  *)
    echo "Usage: $0 {control50|control50_frozen|pool200|pool1000|contrastive|sweep|grid TAG [pool]}"
    exit 1
    ;;
esac
