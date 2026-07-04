#!/usr/bin/env bash
# Push unified hf_release code to HF main (weights unchanged). Run on RTX4090 after dual-verify.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Dual verify (round + legacy main) ==="
( cd hf_release && bash verify.sh && bash verify_legacy.sh )

echo "=== Code-only push to HF main ==="
uv run python scripts/push_qllm_hf.py --revision main \
  --only modeling_qllm.py run_chat.py README.md verify.sh verify_legacy.sh PUSH_TO_HF.md

echo "Done. Legacy users: huggingface-cli download ... --revision main"
