#!/usr/bin/env bash
# V11 PAM math probes — no checkpoint required.
# Runs correctness selftest + full pam_math battery, writes JSON to logs/v11/pam_math/.
#
# Usage:
#   ./scripts/run_v11_pam_math.sh
#   ./scripts/run_v11_pam_math.sh binding   # single test

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh 2>/dev/null || true

PYTHON="${PYTHON_BIN:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-logs/v11/pam_math}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  V11 PAM math probes (Phase A — no checkpoint)"
echo "  Output: $OUT_DIR"
echo "============================================================"

echo ""
echo "--- A1: v11.selftest (train == recurrent) ---"
"$PYTHON" -m v11.selftest

if [[ "${SKIP_FLASH_BENCH:-0}" == "1" ]]; then
  echo "(Skipping bench_flash_pam — SKIP_FLASH_BENCH=1)"
elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo ""
  echo "--- Optional: v11.bench_flash_pam correctness ---"
  "$PYTHON" -m v11.bench_flash_pam --B 2 --T 2048 || true
fi

echo ""
if [[ -n "${1:-}" ]]; then
  echo "--- Running single test: $1 ---"
  "$PYTHON" -m v11.pam_math --test "$1" --output-dir "$OUT_DIR"
else
  echo "--- Running full pam_math battery ---"
  "$PYTHON" -m v11.pam_math --all --output-dir "$OUT_DIR"
  echo ""
  echo "--- Long-context extension (256K; set MAX_DISTANCE=1048576 for 1M) ---"
  MAX_DISTANCE="${MAX_DISTANCE:-262144}"
  "$PYTHON" -m v11.pam_math --test long-context --max-distance "$MAX_DISTANCE" --output-dir "$OUT_DIR"
  echo ""
  echo "--- Language / real-text probes (A10–A11) ---"
  FILLER_TOKENS="${FILLER_TOKENS:-10000}"
  TEXT_TOKENS="${TEXT_TOKENS:-50000}"
  "$PYTHON" -m v11.pam_math_language --test both \
    --filler-tokens "$FILLER_TOKENS" --text-tokens "$TEXT_TOKENS" --output-dir "$OUT_DIR"
fi

echo ""
echo "Done. See $OUT_DIR/"
