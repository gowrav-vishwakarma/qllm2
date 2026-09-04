#!/usr/bin/env bash
# v13_sempty streaming-pretrain launcher (Phase 3: scale the DATA, arch fixed).
#
# Default = the 2026-09-04 scale run: real-102M `baseline_real_pm` + CHRONO +
# OUT_GATE (the 22.96 WikiText reference arch) on a live-streamed blend
#   dclm-edu 45 % / fineweb-edu 42 % / smoltalk2 Mid (ChatML text) 10 % /
#   synthetic recall 3 %      (web-only for the first BLEND_WARMUP tokens)
# for TARGET_TOKENS at B=18 T=2048 (36,864 tok/step), dropout 0, lr 2e-4 cosine.
# Primary metric = streaming holdout val (5 % of the corpus); the WikiText-103
# val is logged as a secondary anchor (NOT comparable 1:1 with the 22.96
# WikiText-only runs - different data, single pass).
#
# Usage (RTX Pro 6000, ALWAYS in tmux, env INSIDE the command string):
#   tmux new-session -d -s sempty_mix \
#     "TAG=mix3b_chrono_gate bash v13_sempty/tmp_pretrain_mix.sh"
# Requires the `sempyt` .pth in the venv and network access to HF (public
# datasets, no token needed).
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
# Run from a private temp copy so editing this file mid-run cannot corrupt the
# wrapper (bash reads scripts incrementally; see tmp_wikitext_fair.sh).
if [ -z "${_MIX_COPY:-}" ]; then
  _copy=$(mktemp /tmp/tmp_pretrain_mix.XXXXXX.sh) && cp "${BASH_SOURCE[0]}" "$_copy" \
    && REPO_ROOT="$REPO_ROOT" _MIX_COPY=1 exec bash "$_copy" "$@"
fi
trap 'rm -f "${BASH_SOURCE[0]}"' EXIT

PRESET="${PRESET:-baseline_real_pm}"
TAG="${TAG:-mix3b_chrono_gate}"
SOURCES="${SOURCES:-dclm,fineweb,smoltalk2_mid,recall}"
WEIGHTS="${WEIGHTS:-0.45,0.42,0.10,0.03}"
TARGET_TOKENS="${TARGET_TOKENS:-3000000000}"   # 3.0B -> 81,380 steps at 18x2048
BLEND_WARMUP="${BLEND_WARMUP:-300000000}"      # web-only for the first 300M tok
BATCH="${BATCH:-18}"
SEQ="${SEQ:-2048}"
WARMUP="${WARMUP:-1000}"
LR="${LR:-2e-4}"
DROPOUT="${DROPOUT:-0.0}"
GRAD_CKPT="${GRAD_CKPT:-0}"    # 96 GB box: off (+23 % tok/s); 1 fits the 4090
CHRONO="${CHRONO:-1}"
OUT_GATE="${OUT_GATE:-1}"
VAL_EVERY="${VAL_EVERY:-2000}"
SAVE_EVERY="${SAVE_EVERY:-4000}"

HASH=$(git rev-parse --short HEAD)
STAMP=$(date +%Y%m%d_%H%M)
LOG="logs/v13_sempty_${TAG}_${HASH}_${STAMP}.log"
echo "=== ${TAG} start $(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$HASH log=$LOG ===" | tee "$LOG"
EXTRA=()
if [ "$GRAD_CKPT" = "1" ]; then EXTRA+=(--gradient_checkpointing); fi
if [ "$CHRONO" = "1" ]; then EXTRA+=(--chrono); fi
if [ "$OUT_GATE" = "1" ]; then EXTRA+=(--out_gate); fi
.venv/bin/python -u -m v13_sempty.train \
  --preset "$PRESET" \
  --dataset mix \
  --pretrain_sources "$SOURCES" \
  --pretrain_weights "$WEIGHTS" \
  --target_tokens "$TARGET_TOKENS" \
  --blend_warmup_tokens "$BLEND_WARMUP" \
  --device cuda \
  --batch_size "$BATCH" \
  --seq_len "$SEQ" \
  --lr "$LR" \
  --warmup_steps "$WARMUP" \
  --dropout "$DROPOUT" \
  --amp_dtype bf16 \
  --fused_ce \
  --fused_pam \
  --ce_gemm_dtype auto \
  --seed 42 \
  --log_interval 50 \
  --val_every "$VAL_EVERY" \
  --gen_every 4000 \
  --gen_max_tokens 80 \
  --save_every_steps "$SAVE_EVERY" \
  --diag_every 2000 \
  --max_val_batches 0 \
  "${EXTRA[@]}" \
  --checkpoint_dir "checkpoints_v13_sempty/${TAG}_${HASH}" >> "$LOG" 2>&1
RC=$?
echo "=== ${TAG} end $(date +%s) exit=$RC ===" | tee -a "$LOG"
echo "exit=$RC" > "${LOG%.log}.exit"
exit "$RC"
