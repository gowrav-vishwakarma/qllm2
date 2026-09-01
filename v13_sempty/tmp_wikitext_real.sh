#!/usr/bin/env bash
# Param-matched real-101M on full WikiText-103, one epoch (dataset-exhausted),
# same verified recipe as the tinystories A/B runs. Compared against the
# existing v13 100.6M complex (NLL 4.36 @100M tok) and v11 ~100M complex
# (val PPL 45.6 @10 epochs) references.
set -uo pipefail
cd /home/gowrav/Development/qllm2
echo "=== wikitext_real start $(date -u +%Y-%m-%dT%H:%M:%SZ) epoch=$(date +%s) ==="
.venv/bin/python -m v13_sempty.train \
  --preset baseline_real_pm \
  --dataset wikitext103 \
  --device cuda \
  --batch_size 8 \
  --seq_len 256 \
  --steps 200000 \
  --max_samples 0 \
  --lr 5e-5 \
  --warmup_steps 100 \
  --amp_dtype bf16 \
  --fused_ce \
  --gradient_checkpointing \
  --seed 42 \
  --log_interval 50 \
  --val_every 2000 \
  --gen_every 5000 \
  --gen_max_tokens 80 \
  --save_every_steps 5000 \
  --diag_every 2000 \
  --max_val_batches 128 \
  --checkpoint_dir checkpoints_v13_sempty/ab_real_wikitext 2>&1 | tee logs/ab_real_wikitext.log
RC=${PIPESTATUS[0]}
echo "=== wikitext_real end epoch=$(date +%s) exit=$RC ==="
echo "exit=$RC" > logs/ab_real_wikitext.exit
exit "$RC"
