#!/usr/bin/env bash
# Run V6 training on WikiText-103 (RTX 4090).
#
# Baseline run: banks + SSM on real entity-rich data (no memory).
# Compare against published baselines (LSTM ~48 PPL, Transformer-XL ~18 PPL at ~30M params).
#
# Usage:
#   ./scripts/run_v6_wikitext103.sh                              # baseline (no memory)
#   ./scripts/run_v6_wikitext103.sh --wm_slots 8                 # small working memory
#   ./scripts/run_v6_wikitext103.sh --wm_slots 16 --im_slots 8   # incremental test

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_ARGS="--dataset wikitext103 --size small-matched --max_samples 9999999 --seq_len 512 --batch_size 14 --epochs 20 --init_seed 42 --gen_every 5000 --gen_prompt 'The history of' --compile --compile_mode reduce-overhead --amp_dtype auto --num_workers 4"

LOG_DIR=$(make_log_dir "v6" "wikitext103_small_matched")
echo "[v6-run] Log directory: $LOG_DIR"

write_run_info "$LOG_DIR" "V6 small-matched on WikiText-103 (RTX 4090)" "$TRAIN_ARGS $*"

CHECKPOINT_DIR="checkpoints_v6_wikitext103"
if echo "$@" | grep -q -- '--resume'; then
  echo "[v6-run] Resuming -- keeping existing checkpoints in $CHECKPOINT_DIR/"
else
  if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
    echo "[v6-run] Fresh start -- clearing old checkpoints in $CHECKPOINT_DIR/"
    rm -rf "$CHECKPOINT_DIR"
  fi
fi

eval "$PYTHON_BIN -m v6.train" \
  $TRAIN_ARGS \
  --log_dir "$LOG_DIR" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  "$@"
