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
# Manual resume after the box itself went down (same commit checked out; pass
# CKPT_DIR/RESUME_LOG explicitly if HEAD moved so the run keeps its dir+log):
#   tmux new-session -d -s sempty_mix \
#     "TAG=mix3b_chrono_gate CKPT_DIR=checkpoints_v13_sempty/mix3b_chrono_gate_<hash> \
#      RESUME_LOG=logs/v13_sempty_mix3b_chrono_gate_<hash>_<stamp>.log \
#      bash v13_sempty/tmp_pretrain_mix.sh"
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
# Retention ladder (one variable per run, stacked on chrono+gate; compare the
# recall-horizon curve, holdout PPL is the guard):
DT_SPREAD="${DT_SPREAD:-0}"    # R1 per-head decay-bias ladder (0 = reference)
NSTATES="${NSTATES:-0}"        # A2 states per head (0 = preset default 1)
VAULT="${VAULT:-0}"            # A2b pinned state + protect gate (needs NSTATES>=2)
DELTA="${DELTA:-0}"            # A3 delta erase/write
GEN_EVERY="${GEN_EVERY:-4000}" # set 0 for vault/delta (no decode path yet)
VAL_EVERY="${VAL_EVERY:-2000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"       # latest.pt (full resume state) every ~7 min
KEEP_EVERY="${KEEP_EVERY:-10000}"      # milestone step_XXXXXX.pt copies (~1.2 GB each)
KEEP_LAST="${KEEP_LAST:-1}"            # rolling: only the newest KEEP_LAST milestones survive
                                       # (latest.pt is overwritten atomically; best_model.pt kept)
MAX_RETRIES="${MAX_RETRIES:-5}"        # auto-resume attempts after a crash
CKPT_DIR="${CKPT_DIR:-}"               # default checkpoints_v13_sempty/<TAG>_<HASH>
RESUME_LOG="${RESUME_LOG:-}"           # append to an existing run log instead of a new one

HASH=$(git rev-parse --short HEAD)
STAMP=$(date +%Y%m%d_%H%M)
CKPT_DIR="${CKPT_DIR:-checkpoints_v13_sempty/${TAG}_${HASH}}"
if [ -n "$RESUME_LOG" ]; then LOG="$RESUME_LOG"; else LOG="logs/v13_sempty_${TAG}_${HASH}_${STAMP}.log"; fi
echo "=== ${TAG} start $(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$HASH log=$LOG ckpt=$CKPT_DIR ===" | tee -a "$LOG"
EXTRA=()
if [ "$GRAD_CKPT" = "1" ]; then EXTRA+=(--gradient_checkpointing); fi
if [ "$CHRONO" = "1" ]; then EXTRA+=(--chrono); fi
if [ "$OUT_GATE" = "1" ]; then EXTRA+=(--out_gate); fi
if [ "$DT_SPREAD" != "0" ]; then EXTRA+=(--dt_spread "$DT_SPREAD"); fi
if [ "$NSTATES" != "0" ]; then EXTRA+=(--n_states "$NSTATES"); fi
if [ "$VAULT" = "1" ]; then EXTRA+=(--vault); fi
if [ "$DELTA" = "1" ]; then EXTRA+=(--delta); fi

# Auto-resume loop: --resume auto picks up <CKPT_DIR>/latest.pt when it exists
# (fresh start otherwise), restoring model/optimizer/LR schedule/step/tokens/
# best-val/RNG and the data-stream cursor. A crash (OOM, HF hiccup, box
# reboot mid-tmux) therefore costs at most SAVE_EVERY steps, not the run.
attempt=0
while :; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 1 ]; then
    echo "=== ${TAG} resume attempt $attempt $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  fi
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
  --gen_every "$GEN_EVERY" \
  --gen_max_tokens 80 \
  --save_every_steps "$SAVE_EVERY" \
  --keep_every_steps "$KEEP_EVERY" \
  --keep_last "$KEEP_LAST" \
  --diag_every 2000 \
  --max_val_batches 0 \
  --resume auto \
  "${EXTRA[@]}" \
  --checkpoint_dir "$CKPT_DIR" >> "$LOG" 2>&1
  RC=$?
  if [ "$RC" -eq 0 ]; then break; fi
  echo "=== ${TAG} attempt $attempt died exit=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  if [ "$attempt" -ge "$MAX_RETRIES" ]; then break; fi
  # A non-finite loss is a real bug, not a transient: do not resume into it.
  if tail -n 40 "$LOG" | grep -q "non-finite loss"; then
    echo "=== ${TAG} non-finite loss: not resuming ===" | tee -a "$LOG"; break
  fi
  sleep 60
done
echo "=== ${TAG} end $(date +%s) exit=$RC attempts=$attempt ===" | tee -a "$LOG"
echo "exit=$RC" > "${LOG%.log}.exit"
exit "$RC"
