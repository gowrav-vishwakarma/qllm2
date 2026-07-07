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

OUT_DIR="${OUT_DIR:-checkpoints_v11_duplex_tokenizer}"
LANGUAGES="${LANGUAGES:-hindi,gujarati}"
N_PER_LANG="${N_PER_LANG:-30000}"
N_ENGLISH="${N_ENGLISH:-30000}"
N_TEXT="${N_TEXT:-32000}"
N_CODEBOOKS="${N_CODEBOOKS:-4}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-2048}"
MODEL_TYPE="${MODEL_TYPE:-unigram}"

uv run python -m v11.duplex.tokenizer \
  --out_dir "$OUT_DIR" \
  --languages "$LANGUAGES" \
  --n_per_lang "$N_PER_LANG" \
  --n_english "$N_ENGLISH" \
  --n_text "$N_TEXT" \
  --n_codebooks "$N_CODEBOOKS" \
  --codebook_size "$CODEBOOK_SIZE" \
  --model_type "$MODEL_TYPE" \
  "$@"

echo "Tokenizer written to ${OUT_DIR}"
