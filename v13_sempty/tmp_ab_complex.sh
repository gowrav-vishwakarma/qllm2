#!/usr/bin/env bash
# A/B arm: complex-384 (preset baseline), matched recipe with the real arm.
set -euo pipefail
cd /home/gowrav/Development/qllm2
exec .venv/bin/python -m v13_sempty.train \
  --preset baseline \
  --dataset tinystories \
  --device cuda \
  --batch_size 8 \
  --seq_len 256 \
  --steps 8000 \
  --lr 5e-5 \
  --warmup_steps 100 \
  --amp_dtype bf16 \
  --fused_ce \
  --gradient_checkpointing \
  --max_samples 20000 \
  --seed 42 \
  --checkpoint_dir checkpoints_v13_sempty/ab_complex
