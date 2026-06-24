#!/usr/bin/env bash
# Stage 1: frozen Whisper + LibriSpeech duplex pairs (open fallback).
# Primary plan dataset is Fisher LDC97S62 (needs LDC license) — use when available.
#
# Persistent run (survives disconnect):
#   tmux new-session -d -s v11_duplex_s1 './scripts/run_v11_duplex_stage1.sh'
#   tmux attach -t v11_duplex_s1
#
# Or nohup:
#   nohup ./scripts/run_v11_duplex_stage1.sh > logs/v11/duplex_stage1_nohup.log 2>&1 &

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh 2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRESET="${PRESET:-duplex_5m}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_PAIRS="${N_PAIRS:-800}"
DATASET="${DATASET:-librispeech}"
LOG_DIR="${LOG_DIR:-logs/v11}"
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/duplex_stage1_${PRESET}_${STAMP}.log"

echo "=== V11 duplex Stage 1 ===" | tee "$LOG_FILE"
echo "preset=$PRESET dataset=$DATASET (Fisher=LDC97S62 when licensed)" | tee -a "$LOG_FILE"
echo "log=$LOG_FILE" | tee -a "$LOG_FILE"

uv run python -m v11.duplex.selftest 2>&1 | tee -a "$LOG_FILE"

uv run python -m v11.duplex.train_stage1 \
  --preset "$PRESET" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --n_pairs "$N_PAIRS" \
  --dataset "$DATASET" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

echo "Done. Checkpoint: checkpoints_v11_${PRESET}_stage1/best_model.pt" | tee -a "$LOG_FILE"
