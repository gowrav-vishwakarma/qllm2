#!/usr/bin/env bash
# Exact v5 small Blackwell training configuration matching the current run.
# Use this when you want the same small-model, high-throughput setup again.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v5/train.py ]] || cd ..

exec ./scripts/run_v5_blackwell.sh \
  --size small \
  --batch_size 64 \
  --seq_len 512 \
  --window_size 512 \
  --attention_backend xformers \
  "$@"
