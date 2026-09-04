#!/usr/bin/env bash
# Apples-to-apples WikiText-103 run for the quality program: T=2048, B=18,
# 10 epochs (3213 steps/epoch = 32130 steps) -- the exact geometry of the
# reference bars (transformer 100.3M -> 22.69, v11 E3-K3 -> 25.77, both 10 ep
# B18 T2048). Fused PAM + fused CE, bf16. Gradient checkpointing is ON so the
# reference batch (B18) fits: without it the 16-block retained activations need
# ~30 GiB at B18 T2048; with it, ~5 GiB at 45.5k tok/s (~7.2 h). Matching the
# reference batch matters more for a fair PPL comparison than the ~30% ckpt
# cost.
#
# Override the arm/geometry via env for the complex A/B (Phase 2):
#   PRESET=baseline TAG=wikitext_complex_fair bash v13_sempty/tmp_wikitext_fair.sh
set -uo pipefail
# repo-root relative to THIS script (v13_sempty/..), so it works on the local
# 4090 box and the remote RTX Pro 6000 (qllm-private) without editing.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PRESET="${PRESET:-baseline_real_pm}"
TAG="${TAG:-wikitext_real_fair}"
BATCH="${BATCH:-18}"
SEQ="${SEQ:-2048}"
STEPS="${STEPS:-32130}"       # 3213 * 10 epochs
WARMUP="${WARMUP:-500}"
LR="${LR:-1e-4}"
RECALL_FRAC="${RECALL_FRAC:-0}"
GRAD_CKPT="${GRAD_CKPT:-1}"    # 1 = fits the 4090; use GRAD_CKPT=0 on the 96 GB box
CHRONO="${CHRONO:-0}"          # N1 Chrono-PAM content-modulated rotary (real arm)
OUT_GATE="${OUT_GATE:-0}"      # N4 per-head content-dependent read-out gate
# Architecture-ladder knobs (real arm; one variable per run, stack on CHRONO=1):
SHORT_CONV="${SHORT_CONV:-0}"  # A1 depthwise causal conv on qkv
NSTATES="${NSTATES:-0}"        # A2 states per head (0 = preset default 1)
VAULT="${VAULT:-0}"            # A2b pinned state + protect gate
DELTA="${DELTA:-0}"            # A3 delta erase/write
COND_MEM="${COND_MEM:-0}"      # A4 conditional n-gram memory

HASH=$(git rev-parse --short HEAD)
STAMP=$(date +%Y%m%d_%H%M)
LOG="logs/v13_sempty_${TAG}_${HASH}_${STAMP}.log"
echo "=== ${TAG} start $(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$HASH log=$LOG ===" | tee "$LOG"
EXTRA=()
if [ "$RECALL_FRAC" != "0" ]; then EXTRA+=(--recall_frac "$RECALL_FRAC"); fi
if [ "$GRAD_CKPT" = "1" ]; then EXTRA+=(--gradient_checkpointing); fi
if [ "$CHRONO" = "1" ]; then EXTRA+=(--chrono); fi
if [ "$OUT_GATE" = "1" ]; then EXTRA+=(--out_gate); fi
if [ "$SHORT_CONV" = "1" ]; then EXTRA+=(--short_conv); fi
if [ "$NSTATES" != "0" ]; then EXTRA+=(--n_states "$NSTATES"); fi
if [ "$VAULT" = "1" ]; then EXTRA+=(--vault); fi
if [ "$DELTA" = "1" ]; then EXTRA+=(--delta); fi
if [ "$COND_MEM" = "1" ]; then EXTRA+=(--cond_mem); fi
.venv/bin/python -u -m v13_sempty.train \
  --preset "$PRESET" \
  --dataset wikitext103 \
  --device cuda \
  --batch_size "$BATCH" \
  --seq_len "$SEQ" \
  --steps "$STEPS" \
  --max_samples 0 \
  --lr "$LR" \
  --warmup_steps "$WARMUP" \
  --amp_dtype bf16 \
  --fused_ce \
  --fused_pam \
  --ce_gemm_dtype auto \
  --seed 42 \
  --log_interval 50 \
  --val_every 2000 \
  --gen_every 8000 \
  --gen_max_tokens 80 \
  --save_every_steps 4000 \
  --diag_every 2000 \
  --max_val_batches 0 \
  "${EXTRA[@]}" \
  --checkpoint_dir "checkpoints_v13_sempty/${TAG}_${HASH}" >> "$LOG" 2>&1
RC=$?
echo "=== ${TAG} end $(date +%s) exit=$RC ===" | tee -a "$LOG"
echo "exit=$RC" > "${LOG%.log}.exit"
exit "$RC"
