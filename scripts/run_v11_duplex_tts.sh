#!/usr/bin/env bash
# Stage B: TTS / T2S training (text -> Mimi codec tokens) with both-directions
# data reuse. Gate: round-trip WER (add --eval_round_trip). Warm-start from the
# Stage A ASR checkpoint (same backbone + vocab).
#
#   tmux new-session -d -s duplex_tts './scripts/run_v11_duplex_tts.sh'
# Resume:
#   RESUME=checkpoints_v11_duplex_100m_tts/latest.pt ./scripts/run_v11_duplex_tts.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRESET="${PRESET:-duplex_100m}"
TOKENIZER_DIR="${TOKENIZER_DIR:-checkpoints_v11_duplex_tokenizer}"
TASK="${TASK:-both}"                 # t2s | both | roundtrip(ablation only)
LANGUAGES="${LANGUAGES:-hindi,gujarati}"
N_PER_LANG="${N_PER_LANG:-3000}"
N_ENGLISH="${N_ENGLISH:-3000}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
INIT_FROM="${INIT_FROM:-checkpoints_v11_${PRESET}_asr/best_model.pt}"
CKPT_DIR="${CKPT_DIR:-checkpoints_v11_${PRESET}_tts}"
RESUME="${RESUME:-}"
LOG_DIR="${LOG_DIR:-logs/v11}"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/duplex_tts_${PRESET}_${TASK}_${STAMP}.log"

if [[ ! -f "${TOKENIZER_DIR}/duplex_spm.model" ]]; then
  echo "Tokenizer not found in ${TOKENIZER_DIR}. Run ./scripts/run_v11_duplex_tokenizer.sh first." | tee "$LOG_FILE"
  exit 1
fi

INIT_ARGS=()
[[ -n "$INIT_FROM" && -f "$INIT_FROM" ]] && INIT_ARGS=(--init_from "$INIT_FROM")
RESUME_ARGS=()
[[ -n "$RESUME" ]] && RESUME_ARGS=(--resume "$RESUME")

echo "=== V11 duplex Stage B (TTS/T2S) task=$TASK ===" | tee "$LOG_FILE"
echo "preset=$PRESET init_from=${INIT_FROM:-none} log=$LOG_FILE" | tee -a "$LOG_FILE"

uv run python -m v11.duplex.train_tts \
  --preset "$PRESET" \
  --tokenizer_dir "$TOKENIZER_DIR" \
  --task "$TASK" \
  --languages "$LANGUAGES" \
  --n_per_lang "$N_PER_LANG" \
  --n_english "$N_ENGLISH" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --ckpt_dir "$CKPT_DIR" \
  "${INIT_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

echo "Done. Checkpoints: ${CKPT_DIR}/best_model.pt + latest.pt" | tee -a "$LOG_FILE"
