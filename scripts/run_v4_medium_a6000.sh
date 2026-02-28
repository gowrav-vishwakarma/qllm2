#!/usr/bin/env bash
# Run v4 medium training on A6000 (48GB) WITH compile. Use inside tmux so it survives disconnect.
#
# Workflow: Find best batch size first (no compile), then run full training with compile.
#   1. ./scripts/tune_batch_a6000.sh --batch_size 16   # then try 24, 32... (1 epoch each, no compile)
#   2. After one epoch or when stable, kill (Ctrl+C). Note max batch size that didn't OOM.
#   3. Full run: ./scripts/run_v4_medium_a6000.sh --batch_size 24   # use your chosen size
#
# On server:
#   tmux new -s v4train
#   cd ~/qllm && ./scripts/run_v4_medium_a6000.sh [--batch_size N]
#   # Detach: Ctrl+B, D. Reattach: tmux attach -t v4train
#
# Optional: log to file
#   ./scripts/run_v4_medium_a6000.sh 2>&1 | tee logs/v4_medium_$(date +%Y%m%d_%H%M).log
#
# Note: With --compile, the first run spends 10-30+ min in torch.compile (CPU-bound).
# GPU stays ~1GB and 0% until compile finishes; then training starts and GPU spikes. Normal.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v4/train_real.py ]] || cd ..
export PATH="$HOME/.local/bin:$PATH"

# remove all cache files
rm -rf .cache/v4_tokens

mkdir -p .cache/v4_tokens checkpoints_v4_real
mkdir -p logs 2>/dev/null || true

uv run python v4/train_real.py \
  --dataset tinystories \
  --size medium \
  --max_length 256 \
  --batch_size 48 \
  --accumulation_steps 1 \
  --epochs 50 \
  --compile \
  --compile_mode reduce-overhead \
  --num_workers 4 \
  --no_metrics \
  --cache_dir .cache/v4_tokens \
  --checkpoint_dir checkpoints_v4_real \
  "$@"
