#!/usr/bin/env bash
# Equal-data 3B run for any architecture (EXPERIMENTS_SEMPY "Equal-data
# scoreboard"): the EXACT Phase 3b data recipe through the same trainer, loss
# and evals, so PAM, Mamba-2 and Transformer++ differ only in the mixer and
# their own screened LR. Generalises tmp_mamba2_baseline.sh (the Mamba-2 run
# @b36e3a5 used that script; it is in git history).
#
#   ARCH=transformer tmux new-session -d -s tf3b "bash v13_sempty/tmp_equal_3b.sh"
#   ARCH=pam LR_FULL=1.2e-3 EXTRA="COUPLE=1" TAG_FULL=mix3b_c1_lr1.2e-3 ...
#
# Chained, no human in the loop:
#   A. LR screen (skipped if LR_FULL is set): tmp_scratch_screen.sh arms
#      2e-4 / 6e-4 / 1.2e-3 (2,700 steps, ~100M tok, real mix from step 0,
#      probes per arm). Lowest final holdout val_loss wins; a higher LR must
#      beat a lower one by > 0.01 nats. Choice saved -> stable on resume.
#   B. Full run: Phase 3b recipe (3B tok, B18 T2048, warmup 1000, 300M-token
#      web-only blend warmup, 0.44/0.41/0.10/0.05, ANSWER_W=100, seed 42).
#   C. 40-trial recall probe + PG-19 per-position loss on best_model.pt.
#
# Env: ARCH pam|mamba2|transformer; LR_FULL; EXTRA="VAR=val ..." (all runs,
# e.g. COUPLE=1 for PAM); TAG_FULL (default mix3b_<arch>_answ100).
# RESUMABLE / idempotent: re-run with SCREEN_HASH=<hash printed at the first
# launch>. Finished parts are skipped, a partial run resumes from latest.pt.
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
if [ -z "${_E3_COPY:-}" ]; then
  _copy=$(mktemp /tmp/tmp_equal_3b.XXXXXX.sh) && cp "${BASH_SOURCE[0]}" "$_copy" \
    && REPO_ROOT="$REPO_ROOT" _E3_COPY=1 exec bash "$_copy" "$@"
fi
# Capture the temp copy's path NOW: a trap body evaluates BASH_SOURCE when it fires, and inside a sourced file that refuses
# (scripts/require_clean_tree.sh: exit 3) it named the guard itself, which the old trap deleted (Lead 2026-11-18). Only /tmp copies.
_self="${BASH_SOURCE[0]}"; trap 'case "$_self" in /tmp/tmp_*.sh) rm -f "$_self";; esac' EXIT

ARCH="${ARCH:?set ARCH=pam|mamba2|transformer}"
case "$ARCH" in
  pam)         BASE_ENV=(CHRONO=1 OUT_GATE=1 DELTA=1); PREFIX=pamlr ;;
  mamba2)      BASE_ENV=(ARCH=mamba2 CHRONO=0 OUT_GATE=0 DELTA=0); PREFIX=m2lr ;;
  transformer) BASE_ENV=(ARCH=transformer CHRONO=0 OUT_GATE=0 DELTA=0); PREFIX=tflr ;;
  *) echo "unknown ARCH=$ARCH"; exit 1 ;;
esac
read -ra EXTRA_ENV <<< "${EXTRA:-}"
BASE_ENV+=(ANSWER_W=100 SOURCES=dclm,fineweb,smoltalk2_mid,recall WEIGHTS=0.44,0.41,0.10,0.05 "${EXTRA_ENV[@]}")
TAG_FULL="${TAG_FULL:-mix3b_${ARCH}_answ100}"
HASH="${SCREEN_HASH:-$(git rev-parse --short HEAD)}"   # pin on relaunch
[ -f scripts/require_clean_tree.sh ] || { echo "launch guard scripts/require_clean_tree.sh missing: REFUSE (restore: git show HEAD:scripts/require_clean_tree.sh > scripts/require_clean_tree.sh)"; exit 1; }   # fail closed (2026-11-16: the guard was found deleted, so source only warned)
source scripts/require_clean_tree.sh "$HASH"           # committed code == pinned code
DRV_LOG="logs/v13_sempty_${TAG_FULL}_driver_${HASH}.log"
exec > >(tee -a "$DRV_LOG") 2>&1
echo "=== equal-3b $TAG_FULL start $(date -u +%FT%TZ) SCREEN_HASH=$HASH arch=$ARCH extra='${EXTRA:-}' ==="

# --- A. LR screen -----------------------------------------------------------
CHOICE="checkpoints_v13_sempty/${TAG_FULL}_lr_choice_${HASH}.txt"
if [ -n "${LR_FULL:-}" ] && [ ! -f "$CHOICE" ]; then
  echo "$LR_FULL" > "$CHOICE"; echo "=== LR given: $LR_FULL (screen skipped) ==="
fi
if [ ! -f "$CHOICE" ]; then
  SCREEN="$PREFIX" SCREEN_HASH="$HASH" BASE="${BASE_ENV[*]}" \
    ARMS="lr2e-4:LR=2e-4 lr6e-4:LR=6e-4 lr1.2e-3:LR=1.2e-3" \
    bash v13_sempty/tmp_scratch_screen.sh
  SDRV="logs/v13_sempty_${PREFIX}_driver_${HASH}.log"
  best_lr=""; best_vl=""
  for lr in 2e-4 6e-4 1.2e-3; do      # ascending LR: a higher LR must win by > 0.01
    vl=$(grep -E "^  lr$lr: [0-9.]+$" "$SDRV" | tail -1 | awk '{print $2}')
    echo "  LR $lr: final holdout val_loss=${vl:-n/a}"
    [ -z "$vl" ] && continue
    if [ -z "$best_vl" ] || awk -v a="$vl" -v b="$best_vl" 'BEGIN{exit !(a < b - 0.01)}'; then
      best_lr="$lr"; best_vl="$vl"
    fi
  done
  [ -z "$best_lr" ] && { echo "LR screen produced no result; stop"; exit 1; }
  echo "$best_lr" > "$CHOICE"; echo "=== LR choice: $best_lr (val_loss $best_vl) ==="
fi
LR_FULL=$(cat "$CHOICE")

# --- B. full Phase 3b-recipe run ----------------------------------------------
CKPT="checkpoints_v13_sempty/${TAG_FULL}_${HASH}"
if [ ! -f "$CKPT/.done" ]; then
  LOG=$(ls -t logs/v13_sempty_${TAG_FULL}_${HASH}_2*.log 2>/dev/null | head -1)
  LOG="${LOG:-logs/v13_sempty_${TAG_FULL}_${HASH}_$(date +%Y%m%d_%H%M).log}"
  echo "=== $TAG_FULL LR=$LR_FULL ckpt=$CKPT log=$LOG $(date -u +%FT%TZ) ==="
  env TAG="$TAG_FULL" CKPT_DIR="$CKPT" RESUME_LOG="$LOG" PIN_HASH="$HASH" MAX_RETRIES=5 \
    "${BASE_ENV[@]}" LR="$LR_FULL" TARGET_TOKENS=3000000000 BLEND_WARMUP=300000000 \
    WARMUP=1000 VAL_EVERY=2000 SAVE_EVERY=1000 KEEP_EVERY=10000 GEN_EVERY=4000 \
    bash v13_sempty/tmp_pretrain_mix.sh
  rc=$?
  echo "=== $TAG_FULL exit=$rc $(date -u +%FT%TZ) ==="
  [ "$rc" -eq 0 ] || { echo "full run failed; stop"; exit 1; }
  touch "$CKPT/.done"
  [ "${PRUNE_LATEST:-1}" = "1" ] && rm -f "$CKPT/latest.pt"   # resume-only; best + milestone stay
fi

# --- C. same evals as every arm ------------------------------------------------
CK="$CKPT/best_model.pt"
PJ="logs/memory_probes/v13_sempty_${TAG_FULL}_${HASH}_t40_behavioral.json"
LJ="logs/pos_loss/v13_sempty_${TAG_FULL}_${HASH}_pg19.json"
[ -f "$PJ" ] || .venv/bin/python scripts/run_memory_behavioral.py --model-type v13_sempty \
  --checkpoint "$CK" --preset baseline_real_pm \
  --context-lengths 128,256,512,1024,2048,4096,8192 --positions 0,0.5,1 \
  --association-counts 1,4,8 --trials 40 --max-context 8192 \
  --output "$PJ" 2>&1 | grep -v scored
[ -f "$LJ" ] || .venv/bin/python scripts/eval_pos_loss.py --model-type v13_sempty \
  --checkpoint "$CK" --context 8192 --docs 50 --output "$LJ"
echo "=== equal-3b $TAG_FULL DONE $(date -u +%FT%TZ) ==="
