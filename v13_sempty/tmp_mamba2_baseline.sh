#!/usr/bin/env bash
# The "wind tunnel" (EXPERIMENTS_SEMPY "Mamba-2 baseline"): reference Mamba-2,
# param-matched (102.6M), trained on the EXACT Phase 3b data recipe through
# the same trainer/loss/evals as PAM, so design and data can be separated.
#
#   tmux new-session -d -s mamba2 "bash v13_sempty/tmp_mamba2_baseline.sh"
#
# Chained, no human in the loop:
#   A. LR screen: 2e-4 (= PAM's LR) / 6e-4 / 1.2e-3, 2,700 steps (~100M tok)
#      each, real mix from step 0. Pick the lowest holdout val_loss at the
#      last step; a higher LR must beat a lower one by > 0.01 nats to win
#      (ties go to the PAM recipe's side). Choice saved -> stable on resume.
#   B. Full run: Phase 3b recipe (3B tok, B18 T2048, warmup 1000, 300M-token
#      web-only blend warmup, 0.44/0.41/0.10/0.05, ANSWER_W=100, seed 42).
#   C. 40-trial recall probe + PG-19 per-position loss on best_model.pt.
#
# RESUMABLE / idempotent: re-run with SCREEN_HASH=<hash printed at the first
# launch>. Finished arms (.done) are skipped, a partial arm resumes from its
# latest.pt into its own log, finished evals are skipped.
# Needs mamba_ssm + causal_conv1d in the venv (v13_sempty/baseline_mamba2.py).
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
if [ -z "${_M2_COPY:-}" ]; then
  _copy=$(mktemp /tmp/tmp_mamba2_baseline.XXXXXX.sh) && cp "${BASH_SOURCE[0]}" "$_copy" \
    && REPO_ROOT="$REPO_ROOT" _M2_COPY=1 exec bash "$_copy" "$@"
fi
trap 'rm -f "${BASH_SOURCE[0]}"' EXIT

HASH="${SCREEN_HASH:-$(git rev-parse --short HEAD)}"   # pin on relaunch
source scripts/require_clean_tree.sh "$HASH"           # committed code == pinned code
DRV_LOG="logs/v13_sempty_mamba2_baseline_driver_${HASH}.log"
exec > >(tee -a "$DRV_LOG") 2>&1
echo "=== mamba2 baseline start $(date -u +%FT%TZ) SCREEN_HASH=$HASH ==="

M2=(ARCH=mamba2 CHRONO=0 OUT_GATE=0 DELTA=0 ANSWER_W=100
    SOURCES=dclm,fineweb,smoltalk2_mid,recall WEIGHTS=0.44,0.41,0.10,0.05)

run_arm() {  # run_arm <tag> <max_retries> VAR=val ... ; returns the launcher rc
  local tag="$1" retries="$2"; shift 2
  local ckpt="checkpoints_v13_sempty/${tag}_${HASH}"
  if [ -f "$ckpt/.done" ]; then echo "=== $tag already done, skip ==="; return 0; fi
  local log
  log=$(ls -t logs/v13_sempty_${tag}_${HASH}_*.log 2>/dev/null | head -1)
  log="${log:-logs/v13_sempty_${tag}_${HASH}_$(date +%Y%m%d_%H%M).log}"
  echo "=== $tag ($*) ckpt=$ckpt log=$log $(date -u +%FT%TZ) ==="
  env TAG="$tag" CKPT_DIR="$ckpt" RESUME_LOG="$log" PIN_HASH="$HASH" \
    MAX_RETRIES="$retries" "${M2[@]}" "$@" bash v13_sempty/tmp_pretrain_mix.sh
  local rc=$?
  echo "=== $tag exit=$rc $(date -u +%FT%TZ) ==="
  [ "$rc" -eq 0 ] && touch "$ckpt/.done"
  return "$rc"
}

# --- A. LR screen -----------------------------------------------------------
CHOICE="checkpoints_v13_sempty/m2lr_choice_${HASH}.txt"
if [ ! -f "$CHOICE" ]; then
  for lr in 2e-4 6e-4 1.2e-3; do
    run_arm "m2lr_${lr}" 1 LR="$lr" TARGET_TOKENS=99532800 BLEND_WARMUP=0 \
      WARMUP=100 VAL_EVERY=900 SAVE_EVERY=900 KEEP_EVERY=0 GEN_EVERY=0 \
      || echo "=== m2lr_${lr} failed (diverged?) -> treated as loss=inf ==="
  done
  best_lr=""; best_vl=""
  for lr in 2e-4 6e-4 1.2e-3; do      # ascending LR: a higher LR must win by >0.01
    log=$(ls -t logs/v13_sempty_m2lr_${lr}_${HASH}_*.log 2>/dev/null | head -1)
    vl=$(grep -o "\[val @ step 2700\] val_loss=[0-9.]*" "$log" 2>/dev/null | tail -1 | sed 's/.*=//')
    echo "  LR $lr: final holdout val_loss=${vl:-n/a}"
    [ -z "$vl" ] && continue
    if [ -z "$best_vl" ] || awk -v a="$vl" -v b="$best_vl" 'BEGIN{exit !(a < b - 0.01)}'; then
      best_lr="$lr"; best_vl="$vl"
    fi
  done
  [ -z "$best_lr" ] && { echo "LR screen produced no result; stop"; exit 1; }
  echo "$best_lr" > "$CHOICE"
  echo "=== LR choice: $best_lr (val_loss $best_vl) ==="
fi
LR_FULL=$(cat "$CHOICE")

# --- B. full Phase 3b-recipe run ----------------------------------------------
TAG_FULL="mix3b_mamba2_answ100"
run_arm "$TAG_FULL" 5 LR="$LR_FULL" TARGET_TOKENS=3000000000 BLEND_WARMUP=300000000 \
  WARMUP=1000 VAL_EVERY=2000 SAVE_EVERY=1000 KEEP_EVERY=10000 GEN_EVERY=4000 \
  || { echo "full run failed; stop"; exit 1; }

# --- C. same evals as PAM ------------------------------------------------------
CK="checkpoints_v13_sempty/${TAG_FULL}_${HASH}/best_model.pt"
PJ="logs/memory_probes/v13_sempty_${TAG_FULL}_${HASH}_t40_behavioral.json"
LJ="logs/pos_loss/v13_sempty_${TAG_FULL}_${HASH}_pg19.json"
[ -f "$PJ" ] || .venv/bin/python scripts/run_memory_behavioral.py --model-type v13_sempty \
  --checkpoint "$CK" --preset baseline_real_pm \
  --context-lengths 128,256,512,1024,2048,4096,8192 --positions 0,0.5,1 \
  --association-counts 1,4,8 --trials 40 --max-context 8192 \
  --output "$PJ" 2>&1 | grep -v scored
[ -f "$LJ" ] || .venv/bin/python scripts/eval_pos_loss.py --model-type v13_sempty \
  --checkpoint "$CK" --context 8192 --docs 50 --output "$LJ"
echo "=== mamba2 baseline DONE $(date -u +%FT%TZ) ==="
