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
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

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
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"

if [[ ! -f "${TOKENIZER_DIR}/duplex_spm.model" ]]; then
  echo "Tokenizer not found in ${TOKENIZER_DIR}. Run ./scripts/run_v11_duplex_tokenizer.sh first."
  exit 1
fi

mkdir -p "$CKPT_DIR"
REUSED_LOG_DIR=0
if [[ -z "${LOG_DIR:-}" && -n "$RESUME" && -f "$LOG_DIR_SIDECAR" ]]; then
  _stored=$(head -n 1 "$LOG_DIR_SIDECAR" | tr -d '\r')
  if [[ -n "$_stored" && -d "$_stored" ]]; then
    LOG_DIR="$_stored"
    REUSED_LOG_DIR=1
  fi
fi
if [[ -z "${LOG_DIR:-}" ]]; then
  LOG_DIR=$(make_log_dir "v11" "duplex_v2_${PRESET}")
fi
mkdir -p "$LOG_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"
LOG_FILE="${LOG_DIR}/duplex_v2_${PRESET}.log"

ARGS=(--preset "$PRESET" --tokenizer_dir "$TOKENIZER_DIR"
      --reply_source "$REPLY_SOURCE" --languages "$LANGUAGES"
      --n_per_lang "$N_PER_LANG" --n_english "$N_ENGLISH"
      --barge_in_prob "$BARGE_IN_PROB" --epochs "$EPOCHS"
      --batch_size "$BATCH_SIZE" --lr "$LR" --ckpt_dir "$CKPT_DIR")
[[ -n "$CONVERSATIONS" ]] && ARGS+=(--conversations "$CONVERSATIONS")
[[ -f "$INIT_ASR" ]] && ARGS+=(--init_asr "$INIT_ASR")
[[ -f "$INIT_TTS" ]] && ARGS+=(--init_tts "$INIT_TTS")
[[ -n "$RESUME" ]] && ARGS+=(--resume "$RESUME")

RUN_DESC="V11 duplex Stage C (unified): preset=$PRESET reply=$REPLY_SOURCE"
if [[ $REUSED_LOG_DIR -eq 1 ]]; then
  append_run_info_resume "$LOG_DIR" "$RUN_DESC (resume)" "${ARGS[*]} $*"
else
  write_run_info "$LOG_DIR" "$RUN_DESC" "${ARGS[*]} $*"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === V11 duplex Stage C (unified interface) ===" | tee "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] preset=$PRESET reply=$REPLY_SOURCE kathbath=$LANGUAGES n_english=$N_ENGLISH init_asr=${INIT_ASR} init_tts=${INIT_TTS}" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] log_dir=$LOG_DIR log=$LOG_FILE ckpt=$CKPT_DIR" | tee -a "$LOG_FILE"

export PYTHONUNBUFFERED=1
uv run python -m v11.duplex.train_duplex_v2 "${ARGS[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"

echo "Done. Log: $LOG_FILE  Checkpoints: ${CKPT_DIR}/best_model.pt + latest.pt" | tee -a "$LOG_FILE"
