#!/usr/bin/env bash
# Runs the param-matched real arm and records the exact exit code so an
# external kill / crash is distinguishable from a clean finish.
set -uo pipefail
cd /home/gowrav/Development/qllm2
echo "=== real_pm wrapper start $(date -u +%Y-%m-%dT%H:%M:%SZ) epoch=$(date +%s) ==="
.venv/bin/python -m v13_sempty.train \
  --preset baseline_real_pm \
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
  --checkpoint_dir checkpoints_v13_sempty/ab_real_pm 2>&1 | tee logs/ab_real_pm.log
RC=${PIPESTATUS[0]}
echo "=== real_pm wrapper end epoch=$(date +%s) exit=$RC ==="
echo "exit=$RC" > logs/ab_real_pm.exit
exit "$RC"
