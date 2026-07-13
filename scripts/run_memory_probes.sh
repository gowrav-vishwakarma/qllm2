#!/usr/bin/env bash
# Memory probes — recurrent matrix-memory evaluation framework.
# Runs correctness selftest + full probe battery, writes JSON to logs/memory_probes/.
#
# Usage:
#   ./scripts/run_memory_probes.sh
#   ./scripts/run_memory_probes.sh binding   # single test

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh 2>/dev/null || true

OUT_DIR="${OUT_DIR:-logs/memory_probes}"
mkdir -p "$OUT_DIR"

run_py() {
  uv run python "$@"
}

echo "============================================================"
echo "  Memory probes (no checkpoint required)"
echo "  Output: $OUT_DIR"
echo "============================================================"

echo ""
echo "--- selftest: train == recurrent ---"
run_py -m memory_probes.selftest

if [[ "${SKIP_FLASH_BENCH:-0}" == "1" ]]; then
  echo "(Skipping bench_flash_pam — SKIP_FLASH_BENCH=1)"
elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo ""
  echo "--- Optional: v11.bench_flash_pam correctness ---"
  run_py -m v11.bench_flash_pam --B 2 --T 2048 || true
fi

echo ""
if [[ -n "${1:-}" ]]; then
  echo "--- Running single test: $1 ---"
  run_py -m memory_probes --test "$1" --output-dir "$OUT_DIR"
else
  echo "--- Running full synthetic battery ---"
  run_py -m memory_probes --all --output-dir "$OUT_DIR"
  echo ""
  echo "--- Long-context extension (256K; set MAX_DISTANCE=1048576 for 1M) ---"
  MAX_DISTANCE="${MAX_DISTANCE:-262144}"
  run_py -m memory_probes --test long-context --max-distance "$MAX_DISTANCE" --output-dir "$OUT_DIR"
  echo ""
  echo "--- Language / real-text probes ---"
  FILLER_TOKENS="${FILLER_TOKENS:-10000}"
  TEXT_TOKENS="${TEXT_TOKENS:-50000}"
  run_py -m memory_probes --test language-filler \
    --filler-tokens "$FILLER_TOKENS" --output-dir "$OUT_DIR"
  run_py -m memory_probes --test rank-text \
    --text-tokens "$TEXT_TOKENS" --output-dir "$OUT_DIR"
fi

echo ""
echo "Done. See $OUT_DIR/"
