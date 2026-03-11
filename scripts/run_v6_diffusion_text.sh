#!/usr/bin/env bash
# V6 Diffusion Text Training on WikiText-103 -- RTX 4090
#
# Baseline run: diffusion denoising objective on real entity-rich data (no memory).
#
# Usage:
#   ./scripts/run_v6_diffusion_text.sh                              # baseline (no memory)
#   ./scripts/run_v6_diffusion_text.sh --wm_slots 16 --im_slots 32  # with memory

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_ARGS="--dataset wikitext103 --size small-matched --mode diffusion_text --max_samples 9999999 --seq_len 512 --batch_size 14 --epochs 10 --diffusion_steps 1000 --noise_schedule cosine --prediction_target x0 --sampling_method ddpm --no_working_memory --no_internal_memory --gen_every 5000 --compile --compile_mode reduce-overhead --amp_dtype auto --num_workers 4"

LOG_DIR=$(make_log_dir "v6" "wikitext103_diffusion_text")
echo "[v6-run] Log directory: $LOG_DIR"

write_run_info "$LOG_DIR" "V6 diffusion text on WikiText-103 (no memory, RTX 4090)" "$TRAIN_ARGS $*"

CHECKPOINT_DIR="checkpoints_v6_wikitext103_diffusion"
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
