#!/usr/bin/env bash
# Compare WikiText vs DCLM-Edu holdout PPL for V11 checkpoints (no training).
#
# Usage:
#   ./scripts/run_v11_eval_checkpoints.sh
#
# Default: wiki baseline (25.77) vs DCLM pretrain ckpt (66.27 on wiki during training).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh

WIKI_CKPT="${WIKI_CKPT:-checkpoints_v11_e3_k3/best_model.pt}"
DCLM_CKPT="${DCLM_CKPT:-checkpoints_v11_e3_k3_dclm/best_model.pt}"

echo "============================================================"
echo "  V11 checkpoint validation (WikiText val + DCLM holdout)"
echo "  Wiki ckpt: $WIKI_CKPT"
echo "  DCLM ckpt: $DCLM_CKPT"
echo "============================================================"

eval "$PYTHON_BIN -m v11.eval_checkpoints" \
  --checkpoints "$WIKI_CKPT" "$DCLM_CKPT" \
  --labels wiki,dclm \
  --batch_size 18 \
  --seq_len 2048 \
  --dclm_holdout_tokens 500000 \
  "$@"
