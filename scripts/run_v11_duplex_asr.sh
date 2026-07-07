#!/usr/bin/env bash
# Stage A: ASR / S2T training for the duplex voice model (frozen Whisper frames
# -> PAM backbone -> transcript in the unified vocab). Gate: held-out CER.
#
# Persistent run:
#   tmux new-session -d -s duplex_asr './scripts/run_v11_duplex_asr.sh'
#   tmux attach -t duplex_asr
# Resume:
#   RESUME=checkpoints_v11_duplex_100m_asr/latest.pt ./scripts/run_v11_duplex_asr.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRESET="${PRESET:-duplex_100m}"
TOKENIZER_DIR="${TOKENIZER_DIR:-checkpoints_v11_duplex_tokenizer}"
LANGUAGES="${LANGUAGES:-hindi,gujarati}"
N_PER_LANG="${N_PER_LANG:-4000}"
N_ENGLISH="${N_ENGLISH:-4000}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
CKPT_DIR="${CKPT_DIR:-checkpoints_v11_${PRESET}_asr}"
RESUME="${RESUME:-}"
LOG_DIR="${LOG_DIR:-logs/v11}"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/duplex_asr_${PRESET}_${STAMP}.log"

if [[ ! -f "${TOKENIZER_DIR}/duplex_spm.model" ]]; then
  echo "Tokenizer not found in ${TOKENIZER_DIR}. Run ./scripts/run_v11_duplex_tokenizer.sh first." | tee "$LOG_FILE"
  exit 1
fi

RESUME_ARGS=()
[[ -n "$RESUME" ]] && RESUME_ARGS=(--resume "$RESUME")

echo "=== V11 duplex Stage A (ASR) ===" | tee "$LOG_FILE"
echo "preset=$PRESET tokenizer=$TOKENIZER_DIR langs=$LANGUAGES log=$LOG_FILE" | tee -a "$LOG_FILE"

uv run python -m v11.duplex.train_asr \
  --preset "$PRESET" \
  --tokenizer_dir "$TOKENIZER_DIR" \
  --languages "$LANGUAGES" \
  --n_per_lang "$N_PER_LANG" \
  --n_english "$N_ENGLISH" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --ckpt_dir "$CKPT_DIR" \
  "${RESUME_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

echo "Done. Checkpoints: ${CKPT_DIR}/best_model.pt (best CER) + latest.pt" | tee -a "$LOG_FILE"
