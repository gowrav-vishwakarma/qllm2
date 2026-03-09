#!/bin/bash
# V6 Diffusion Text Training -- RTX 4090
#
# Uses the same PhaseFieldBackbone as autoregressive, but trains with
# denoising diffusion objective on TinyStories.
#
# Usage:
#   bash scripts/run_v6_diffusion_text.sh
#   bash scripts/run_v6_diffusion_text.sh 2>&1 | tee logs/diffusion_text.log

set -euo pipefail

python -m v6.train \
    --size small-matched \
    --mode diffusion_text \
    --epochs 5 \
    --max_samples 9999999 \
    --batch_size 20 \
    --seq_len 256 \
    --diffusion_steps 1000 \
    --noise_schedule cosine \
    --prediction_target x0 \
    --sampling_method ddpm \
    --wm_slots 16 \
    --im_slots 32 \
    --gen_every 500 \
    --checkpoint_dir checkpoints_v6_diffusion_text \
    --log_dir logs
