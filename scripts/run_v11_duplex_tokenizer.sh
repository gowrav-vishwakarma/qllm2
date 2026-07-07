#!/usr/bin/env bash
# Train the unified duplex tokenizer (hi/gu/en SentencePiece + codec/control map).
#
#   ./scripts/run_v11_duplex_tokenizer.sh
#   OUT_DIR=... LANGUAGES=hindi,gujarati N_PER_LANG=40000 ./scripts/run_v11_duplex_tokenizer.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh 2>/dev/null || true
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

OUT_DIR="${OUT_DIR:-checkpoints_v11_duplex_tokenizer}"
LANGUAGES="${LANGUAGES:-hindi,gujarati}"
N_PER_LANG="${N_PER_LANG:-30000}"
N_ENGLISH="${N_ENGLISH:-30000}"
N_TEXT="${N_TEXT:-32000}"
N_CODEBOOKS="${N_CODEBOOKS:-4}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-2048}"
MODEL_TYPE="${MODEL_TYPE:-unigram}"

if [[ -z "${LOG_DIR:-}" ]]; then
  LOG_DIR=$(make_log_dir "v11" "duplex_tokenizer")
fi
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/duplex_tokenizer.log"

RUN_ARGS="--out_dir $OUT_DIR --languages $LANGUAGES --n_per_lang $N_PER_LANG \
  --n_english $N_ENGLISH --n_text $N_TEXT --n_codebooks $N_CODEBOOKS \
  --codebook_size $CODEBOOK_SIZE --model_type $MODEL_TYPE $*"
write_run_info "$LOG_DIR" "V11 duplex tokenizer (hi/gu/en + codec layout)" "$RUN_ARGS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === V11 duplex tokenizer ===" | tee "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] out_dir=$OUT_DIR languages=$LANGUAGES n_per_lang=$N_PER_LANG n_english=$N_ENGLISH" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] log_dir=$LOG_DIR log=$LOG_FILE" | tee -a "$LOG_FILE"

export PYTHONUNBUFFERED=1
uv run python -m v11.duplex.tokenizer \
  --out_dir "$OUT_DIR" \
  --languages "$LANGUAGES" \
  --n_per_lang "$N_PER_LANG" \
  --n_english "$N_ENGLISH" \
  --n_text "$N_TEXT" \
  --n_codebooks "$N_CODEBOOKS" \
  --codebook_size "$CODEBOOK_SIZE" \
  --model_type "$MODEL_TYPE" \
  "$@" 2>&1 | tee -a "$LOG_FILE"

echo "Tokenizer written to ${OUT_DIR}. Log: $LOG_FILE" | tee -a "$LOG_FILE"
