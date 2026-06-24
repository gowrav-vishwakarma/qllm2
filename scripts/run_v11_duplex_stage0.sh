#!/usr/bin/env bash
# Stage 0: synthetic duplex dynamics on 4090 (or CPU smoke).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

PRESET="${PRESET:-duplex_5m}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EXTRA="${*:-}"

echo "=== V11 duplex Stage 0: preset=$PRESET ==="
uv run python scripts/count_duplex_params.py --preset "$PRESET"
uv run python -m v11.duplex.selftest
uv run python -m v11.duplex.train \
  --preset "$PRESET" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  $EXTRA
uv run python -m v11.duplex.probe \
  --preset "$PRESET" \
  --checkpoint "checkpoints_v11_${PRESET}_stage0/best_model.pt"
