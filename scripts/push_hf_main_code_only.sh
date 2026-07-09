#!/usr/bin/env bash
# Push unified hf_release code to HF main (weights unchanged). Run on RTX4090 after verify.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Verify current round bundle ==="
( cd hf_release && bash verify.sh )

echo "=== Code-only push to HF main ==="
uv run python scripts/push_qllm_hf.py --revision main \
  --only modeling_qllm.py run_chat.py run_complete.py eval_compare.py README.md verify.sh PUSH_TO_HF.md

echo "Done."
