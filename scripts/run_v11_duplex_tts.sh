#!/usr/bin/env bash
# Focused Stage B TTS: text -> Mimi codec tokens. From-scratch (no ASR warm-start).
#
#   tmux new-session -d -s duplex_tts './scripts/run_v11_duplex_tts.sh'
#   BACKBONE=transformer ./scripts/run_v11_duplex_tts.sh
# Resume:
#   RESUME=checkpoints_v11_duplex_100m_tts_pam_t2s/latest.pt ./scripts/run_v11_duplex_tts.sh
#
# Smoke (tiny): ./scripts/run_v11_duplex_tts_smoke.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh 2>/dev/null || true
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

PRESET="${PRESET:-duplex_100m}"
BACKBONE="${BACKBONE:-pam}"          # pam | transformer
TOKENIZER_DIR="${TOKENIZER_DIR:-checkpoints_v11_duplex_tokenizer}"
TASK="${TASK:-t2s}"                  # t2s (default) | both | roundtrip
LANGUAGES="${LANGUAGES:-hindi,gujarati}"
N_PER_LANG="${N_PER_LANG:-2000}"
N_ENGLISH="${N_ENGLISH:-2000}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
INIT_FROM="${INIT_FROM:-}"           # empty = train TTS from scratch
CKPT_DIR="${CKPT_DIR:-checkpoints_v11_${PRESET}_tts_${BACKBONE}_${TASK}}"
CODEC_CACHE="${CODEC_CACHE:-.cache/mimi_codes}"
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
  LOG_DIR=$(make_log_dir "v11" "duplex_tts_${PRESET}_${BACKBONE}_${TASK}")
fi
mkdir -p "$LOG_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"
LOG_FILE="${LOG_DIR}/duplex_tts_${PRESET}_${BACKBONE}_${TASK}.log"

INIT_ARGS=()
[[ -n "$INIT_FROM" && -f "$INIT_FROM" ]] && INIT_ARGS=(--init_from "$INIT_FROM")
RESUME_ARGS=()
[[ -n "$RESUME" ]] && RESUME_ARGS=(--resume "$RESUME")

RUN_ARGS="--preset $PRESET --backbone $BACKBONE --tokenizer_dir $TOKENIZER_DIR --task $TASK \
  --languages $LANGUAGES --n_per_lang $N_PER_LANG --n_english $N_ENGLISH \
  --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --ckpt_dir $CKPT_DIR \
  --codec_cache $CODEC_CACHE ${INIT_ARGS[*]} ${RESUME_ARGS[*]} $*"
RUN_DESC="V11 duplex TTS: backbone=$BACKBONE preset=$PRESET task=$TASK"

if [[ $REUSED_LOG_DIR -eq 1 ]]; then
  append_run_info_resume "$LOG_DIR" "$RUN_DESC (resume)" "$RUN_ARGS"
else
  write_run_info "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === V11 duplex TTS backbone=$BACKBONE task=$TASK ===" | tee "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] preset=$PRESET langs=$LANGUAGES n_per_lang=$N_PER_LANG n_english=$N_ENGLISH init_from=${INIT_FROM:-none}" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] log_dir=$LOG_DIR log=$LOG_FILE ckpt=$CKPT_DIR" | tee -a "$LOG_FILE"

# python -u + PYTHONUNBUFFERED: line-buffered log (tee is a copy, not the only sink)
uv run python -u -m v11.duplex.train_tts \
  --preset "$PRESET" \
  --backbone "$BACKBONE" \
  --tokenizer_dir "$TOKENIZER_DIR" \
  --task "$TASK" \
  --languages "$LANGUAGES" \
  --n_per_lang "$N_PER_LANG" \
  --n_english "$N_ENGLISH" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --ckpt_dir "$CKPT_DIR" \
  --codec_cache "$CODEC_CACHE" \
  "${INIT_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

echo "Done. Log: $LOG_FILE  Checkpoints: ${CKPT_DIR}/best_model.pt + latest.pt" | tee -a "$LOG_FILE"
