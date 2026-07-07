#!/usr/bin/env bash
# Stage C: unified duplex interface model (joint S2T + T2S + control), warm-started
# from Stage A (ASR) + Stage B (TTS) checkpoints.
#
#   tmux new-session -d -s duplex_v2 './scripts/run_v11_duplex_v2.sh'
# Resume:
#   RESUME=checkpoints_v11_duplex_100m_duplex_v2/latest.pt ./scripts/run_v11_duplex_v2.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRESET="${PRESET:-duplex_100m}"
TOKENIZER_DIR="${TOKENIZER_DIR:-checkpoints_v11_duplex_tokenizer}"
REPLY_SOURCE="${REPLY_SOURCE:-corpus}"       # corpus | brain
CONVERSATIONS="${CONVERSATIONS:-}"
LANGUAGES="${LANGUAGES:-hindi,gujarati}"
N_PER_LANG="${N_PER_LANG:-2500}"
N_ENGLISH="${N_ENGLISH:-2500}"
BARGE_IN_PROB="${BARGE_IN_PROB:-0.25}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-6}"
LR="${LR:-2e-4}"
INIT_ASR="${INIT_ASR:-checkpoints_v11_${PRESET}_asr/best_model.pt}"
INIT_TTS="${INIT_TTS:-checkpoints_v11_${PRESET}_tts/best_model.pt}"
CKPT_DIR="${CKPT_DIR:-checkpoints_v11_${PRESET}_duplex_v2}"
RESUME="${RESUME:-}"
LOG_DIR="${LOG_DIR:-logs/v11}"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/duplex_v2_${PRESET}_${STAMP}.log"

if [[ ! -f "${TOKENIZER_DIR}/duplex_spm.model" ]]; then
  echo "Tokenizer not found in ${TOKENIZER_DIR}. Run ./scripts/run_v11_duplex_tokenizer.sh first." | tee "$LOG_FILE"
  exit 1
fi

ARGS=(--preset "$PRESET" --tokenizer_dir "$TOKENIZER_DIR"
      --reply_source "$REPLY_SOURCE" --languages "$LANGUAGES"
      --n_per_lang "$N_PER_LANG" --n_english "$N_ENGLISH"
      --barge_in_prob "$BARGE_IN_PROB" --epochs "$EPOCHS"
      --batch_size "$BATCH_SIZE" --lr "$LR" --ckpt_dir "$CKPT_DIR")
[[ -n "$CONVERSATIONS" ]] && ARGS+=(--conversations "$CONVERSATIONS")
[[ -f "$INIT_ASR" ]] && ARGS+=(--init_asr "$INIT_ASR")
[[ -f "$INIT_TTS" ]] && ARGS+=(--init_tts "$INIT_TTS")
[[ -n "$RESUME" ]] && ARGS+=(--resume "$RESUME")

echo "=== V11 duplex Stage C (unified interface) ===" | tee "$LOG_FILE"
echo "preset=$PRESET reply=$REPLY_SOURCE init_asr=${INIT_ASR} init_tts=${INIT_TTS} log=$LOG_FILE" | tee -a "$LOG_FILE"

uv run python -m v11.duplex.train_duplex_v2 "${ARGS[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"

echo "Done. Checkpoints: ${CKPT_DIR}/best_model.pt + latest.pt" | tee -a "$LOG_FILE"
