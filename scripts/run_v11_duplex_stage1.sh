#!/usr/bin/env bash
# Stage 1: frozen Whisper + Kathbath multilingual (LibriSpeech mix optional).
#
# Persistent run (survives disconnect):
#   tmux new-session -d -s v11_duplex_ml './scripts/run_v11_duplex_stage1.sh'
#   tmux attach -t v11_duplex_ml
#
# Resume after crash:
#   RESUME=checkpoints_v11_duplex_5m_stage1_ml/latest.pt ./scripts/run_v11_duplex_stage1.sh
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
N_PAIRS="${N_PAIRS:-400}"
N_PAIRS_PER_LANG="${N_PAIRS_PER_LANG:-200}"
DATASET="${DATASET:-kathbath}"
LANGUAGES="${LANGUAGES:-hindi,gujarati}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
CKPT_DIR="${CKPT_DIR:-checkpoints_v11_${PRESET}_stage1_ml}"
RESUME="${RESUME:-}"
LOG_DIR="${LOG_DIR:-logs/v11}"
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/duplex_stage1_${PRESET}_${DATASET}_${STAMP}.log"

echo "=== V11 duplex Stage 1 ===" | tee "$LOG_FILE"
echo "preset=$PRESET dataset=$DATASET languages=$LANGUAGES" | tee -a "$LOG_FILE"
echo "ckpt_dir=$CKPT_DIR save_every_steps=$SAVE_EVERY_STEPS resume=${RESUME:-none}" | tee -a "$LOG_FILE"
echo "log=$LOG_FILE" | tee -a "$LOG_FILE"

uv run python -m v11.duplex.selftest 2>&1 | tee -a "$LOG_FILE"

RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

uv run python -m v11.duplex.train_stage1 \
  --preset "$PRESET" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --n_pairs "$N_PAIRS" \
  --n_pairs_per_lang "$N_PAIRS_PER_LANG" \
  --languages "$LANGUAGES" \
  --dataset "$DATASET" \
  --ckpt_dir "$CKPT_DIR" \
  --save_every_steps "$SAVE_EVERY_STEPS" \
  "${RESUME_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

echo "Done. Checkpoints: ${CKPT_DIR}/latest.pt (resume) + best_model.pt" | tee -a "$LOG_FILE"
