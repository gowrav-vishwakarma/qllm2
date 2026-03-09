#!/bin/bash
# V6 Diffusion Image Training -- RTX 4090
#
# Uses the same PhaseFieldBackbone as autoregressive, but trains with
# denoising diffusion on Tiny ImageNet images.
#
# Usage:
#   bash scripts/run_v6_diffusion_image.sh
#   bash scripts/run_v6_diffusion_image.sh 2>&1 | tee logs/diffusion_image.log
#
# Switch encoder:
#   --image_encoder patch  (default, ViT-style, fast)
#   --image_encoder fft    (novel, leverages complex ops)

set -euo pipefail

python -m v6.train \
    --size small-matched \
    --mode diffusion_image \
    --epochs 10 \
    --batch_size 64 \
    --image_size 64 \
    --image_encoder patch \
    --patch_size 8 \
    --image_dataset tiny_imagenet \
    --diffusion_steps 1000 \
    --noise_schedule cosine \
    --prediction_target x0 \
    --sampling_method ddpm \
    --gen_every 500 \
    --checkpoint_dir checkpoints_v6_diffusion_image \
    --log_dir logs
