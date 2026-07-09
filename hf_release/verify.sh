#!/usr/bin/env bash
# Verify hf_release bundle runs end-to-end (no qllm2 repo imports).
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

PROMPT='What is the capital of France?'
echo "=== QLLM HF release verification (SFT chat) ==="
echo "Prompt: $PROMPT"
echo

OUT=$(uv run python run_chat.py \
  --checkpoint qllm_v11_e3k3_chat.pt \
  --prompt "$PROMPT" \
  --no-think \
  --temperature 0.0 \
  --max_new_tokens 64 2>&1) || {
  echo "$OUT"
  echo "FAIL: run_chat.py exited with error"
  exit 1
}

echo "$OUT"

if echo "$OUT" | grep -qi 'paris'; then
  echo "PASS: SFT reply mentions Paris"
else
  echo "FAIL: SFT reply does not mention Paris"
  exit 1
fi

if echo "$OUT" | grep -q 'stopped_on_im_end=True'; then
  echo "PASS: SFT stopped on im_end"
else
  echo "FAIL: SFT did not stop on im_end"
  exit 1
fi

PRETRAIN_PREFIX='The capital of France is'
echo
echo "=== QLLM HF release verification (pretrain completion) ==="
echo "Prefix: $PRETRAIN_PREFIX"
echo

OUT_PT=$(uv run python run_complete.py \
  --checkpoint qllm_v11_e3k3_pretrain.pt \
  --prompt "$PRETRAIN_PREFIX" \
  --temperature 0.1 \
  --max_new_tokens 32 2>&1) || {
  echo "$OUT_PT"
  echo "FAIL: run_complete.py exited with error"
  exit 1
}

echo "$OUT_PT"

if echo "$OUT_PT" | grep -qi 'paris'; then
  echo "PASS: pretrain completion mentions Paris"
else
  echo "FAIL: pretrain completion does not mention Paris"
  exit 1
fi

echo
echo "=== All checks passed ==="
