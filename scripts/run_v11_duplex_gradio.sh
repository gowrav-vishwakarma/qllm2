#!/usr/bin/env bash
# Gradio talk demo for V11 duplex checkpoints.
#
#   ./scripts/run_v11_duplex_gradio.sh
#   ./scripts/run_v11_duplex_gradio.sh --port 7861 --share

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh 2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== V11 Duplex Gradio ==="
echo "Checkpoints: checkpoints_v11_duplex*/best_model.pt | latest.pt"
echo "Open http://127.0.0.1:${PORT:-7860} after launch"

uv run python -m v11.duplex.gradio_app --root "$PWD" "$@"
