#!/usr/bin/env bash
# Verify the SCI-1 release folder runs end to end with torch + transformers only
# (no qllm2 repo, no sempyt). Greedy, so the result is deterministic.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

echo "=== chat: What is the capital of France? ==="
OUT=$("$PY" run_chat.py --prompt 'What is the capital of France?' --max_new_tokens 64 --device cpu 2>&1)
echo "$OUT"
echo "$OUT" | grep -qi 'paris' && echo "PASS: reply mentions Paris" || { echo "FAIL: no Paris"; exit 1; }
echo "$OUT" | grep -q 'stopped_on_im_end=True' && echo "PASS: stopped on <|im_end|>" || { echo "FAIL: no <|im_end|>"; exit 1; }

echo; echo "=== completion: The capital of France is ==="
OUT=$("$PY" run_complete.py --prompt 'The capital of France is' --temperature 0 --max_new_tokens 24 --device cpu 2>&1)
echo "$OUT"
echo "$OUT" | grep -qi 'paris' && echo "PASS: completion mentions Paris" || { echo "FAIL: no Paris"; exit 1; }
echo; echo "=== All checks passed ==="
