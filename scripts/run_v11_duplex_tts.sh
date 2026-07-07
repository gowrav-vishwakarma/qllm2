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
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

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
  LOG_DIR=$(make_log_dir "v11" "duplex_tts_${PRESET}_${TASK}")
fi
mkdir -p "$LOG_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"
LOG_FILE="${LOG_DIR}/duplex_tts_${PRESET}_${TASK}.log"

INIT_ARGS=()
[[ -n "$INIT_FROM" && -f "$INIT_FROM" ]] && INIT_ARGS=(--init_from "$INIT_FROM")
RESUME_ARGS=()
[[ -n "$RESUME" ]] && RESUME_ARGS=(--resume "$RESUME")

RUN_ARGS="--preset $PRESET --tokenizer_dir $TOKENIZER_DIR --task $TASK \
  --languages $LANGUAGES --n_per_lang $N_PER_LANG --n_english $N_ENGLISH \
  --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --ckpt_dir $CKPT_DIR \
  ${INIT_ARGS[*]} ${RESUME_ARGS[*]} $*"
RUN_DESC="V11 duplex Stage B (TTS/T2S): preset=$PRESET task=$TASK"

if [[ $REUSED_LOG_DIR -eq 1 ]]; then
  append_run_info_resume "$LOG_DIR" "$RUN_DESC (resume)" "$RUN_ARGS"
else
  write_run_info "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === V11 duplex Stage B (TTS/T2S) task=$TASK ===" | tee "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] preset=$PRESET kathbath=$LANGUAGES n_per_lang=$N_PER_LANG n_english=$N_ENGLISH init_from=${INIT_FROM:-none}" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] log_dir=$LOG_DIR log=$LOG_FILE ckpt=$CKPT_DIR" | tee -a "$LOG_FILE"

export PYTHONUNBUFFERED=1
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

echo "Done. Log: $LOG_FILE  Checkpoints: ${CKPT_DIR}/best_model.pt + latest.pt" | tee -a "$LOG_FILE"
