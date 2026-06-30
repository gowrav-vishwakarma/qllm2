#!/usr/bin/env bash
# Verify hf_release bundle runs end-to-end (no qllm2 repo imports).
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

PROMPT='What is the capital of France?'
echo "=== QLLM HF release verification ==="
echo "Prompt: $PROMPT"
echo

OUT=$(uv run python run_chat.py \
  --checkpoint qllm_v11_e3k3_chat.pt \
  --prompt "$PROMPT" \
  --temperature 0.0 \
  --max_new_tokens 32 2>&1) || {
  echo "$OUT"
  echo "FAIL: run_chat.py exited with error"
  exit 1
}

echo "$OUT"

if echo "$OUT" | grep -qi 'paris'; then
  echo "PASS: reply mentions Paris"
else
  echo "FAIL: reply does not mention Paris"
  exit 1
fi

if echo "$OUT" | grep -q 'stopped_on_im_end=True'; then
  echo "PASS: stopped on im_end"
else
  echo "FAIL: did not stop on im_end"
  exit 1
fi

echo
echo "=== All checks passed ==="
