#!/usr/bin/env bash
# Cross-architecture smoke test: run the memory probes on Transformer (KV cache)
# and Mamba (HF SSM, slow path) adapters using tiny CPU-friendly configs.
#
# No GPU and no extra dependencies required (mamba_ssm / causal_conv1d NOT needed).
#
# Usage:
#   ./scripts/smoke_test_adapters.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON="${PYTHON_BIN:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-logs/memory_probes/smoke}"
mkdir -p "$OUT_DIR"

# Keep Mamba on CPU and quiet about the missing fast-path kernels.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

run() {
  echo ""
  echo "------------------------------------------------------------"
  echo "  $*"
  echo "------------------------------------------------------------"
  "$@"
}

echo "============================================================"
echo "  Memory probes — Transformer + Mamba smoke test"
echo "  Output: $OUT_DIR"
echo "============================================================"

# Transformer KV cache: associative (binding, persistence, niah) + stateful (rank).
run "$PYTHON" -m memory_probes --arch transformer --test binding \
  --max-n 64 --trials 5 --output-dir "$OUT_DIR"
run "$PYTHON" -m memory_probes --arch transformer --test rank \
  --steps 128 --output-dir "$OUT_DIR"
run "$PYTHON" -m memory_probes --arch transformer --test niah \
  --distances 64,256 --output-dir "$OUT_DIR"

# Mamba SSM (slow path): stateful tier (rank). Associative probes SKIP gracefully.
run "$PYTHON" -m memory_probes --arch mamba --test rank \
  --steps 96 --arch-dim 32 --output-dir "$OUT_DIR"
run "$PYTHON" -m memory_probes --arch mamba --test binding \
  --arch-dim 32 --output-dir "$OUT_DIR"   # expected: SKIP (no native associative read)

echo ""
echo "Smoke test complete. JSON in $OUT_DIR/"
