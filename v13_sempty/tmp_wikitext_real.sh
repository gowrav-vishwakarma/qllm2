#!/usr/bin/env bash
# Param-matched real-101M on full WikiText-103, one epoch, on the fused path:
# Triton PAM scan + Triton fused CE, no gradient checkpointing (peak 7.1 GiB
# at B32 T256, ~67k tok/s -> one epoch (~118M tok) in ~30 min).
#
# Geometry vs the 2026-09-01 run (B8 T256 lr 5e-5, 57.6k steps, val PPL 68.75):
#   B32 T256 = 8192 tok/step (4x; the throughput sweet spot from the grid),
#   lr 1e-4 = 5e-5 * sqrt(4) (sqrt batch scaling), warmup 100 steps,
#   --steps 14400 ~= one epoch so warmup-cosine completes at the epoch end
#   (the earlier run's cosine horizon was 200k steps, i.e. constant lr).
#   val cadence kept at ~4M tokens (every 500 steps).
set -uo pipefail
cd /home/gowrav/Development/qllm2
HASH=$(git rev-parse --short HEAD)
STAMP=$(date +%Y%m%d_%H%M)
LOG="logs/v13_sempty_wikitext_real_${HASH}_${STAMP}.log"
echo "=== wikitext_real start $(date -u +%Y-%m-%dT%H:%M:%SZ) epoch=$(date +%s) commit=$HASH log=$LOG ===" | tee "$LOG"
.venv/bin/python -u -m v13_sempty.train \
  --preset baseline_real_pm \
  --dataset wikitext103 \
  --device cuda \
  --batch_size 32 \
  --seq_len 256 \
  --steps 14400 \
  --max_samples 0 \
  --lr 1e-4 \
  --warmup_steps 100 \
  --amp_dtype bf16 \
  --fused_ce \
  --fused_pam \
  --ce_gemm_dtype auto \
  --seed 42 \
  --log_interval 20 \
  --val_every 500 \
  --gen_every 2500 \
  --gen_max_tokens 80 \
  --save_every_steps 2500 \
  --diag_every 500 \
  --max_val_batches 128 \
  --checkpoint_dir checkpoints_v13_sempty/wikitext_real_fused_${HASH} >> "$LOG" 2>&1
RC=$?
echo "=== wikitext_real end epoch=$(date +%s) exit=$RC ===" | tee -a "$LOG"
echo "exit=$RC" > "${LOG%.log}.exit"
exit "$RC"
