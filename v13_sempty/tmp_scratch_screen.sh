#!/usr/bin/env bash
# From-scratch screen: N arms trained from step 0 on the real Phase 3b mix
# (recall docs from token 0, no web-only warmup), identical except each arm's
# env overrides, run SEQUENTIALLY (never concurrent); then the 40-trial recall
# probe + PG-19 per-position loss for every arm. Same protocol as the Mamba-2
# LR screen (tmp_mamba2_baseline.sh step A): 2,700 steps x 36,864 tok = 99.5M
# tokens, warmup 100, val every 900 (BASE/ARMS may override any of these, e.g.
# TARGET_TOKENS/VAL_EVERY/KEEP_EVERY for a longer run with milestones). Use it
# for LR screens and for architecture changes a warm start cannot judge fairly.
#
#   SCREEN=pamlr ARMS="lr2e-4:LR=2e-4 lr6e-4:LR=6e-4 lr1.2e-3:LR=1.2e-3" \
#     tmux new-session -d -s pamlr "bash v13_sempty/tmp_scratch_screen.sh"
#
# ARMS: space-separated "name[:VAR=val,VAR=val]" passed to tmp_pretrain_mix.sh
# for that arm only. Defaults for every arm = the Phase 3b PAM recipe
# (CHRONO=1 OUT_GATE=1 DELTA=1 ANSWER_W=100, 0.44/0.41/0.10/0.05, seed 42);
# BASE="VAR=val ..." overrides them for all arms (e.g. ARCH=mamba2 ...).
# A diverged arm is reported as failed and the screen goes on.
#
# Disk: a finished arm's latest.pt is deleted (PRUNE_LATEST=0 keeps it).
#
# RESUMABLE / idempotent: re-run the SAME command with SCREEN_HASH=<hash of the
# first launch>. A finished arm (.done) is skipped, a half-done arm resumes
# from its latest.pt into its own log, finished probes are skipped.
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
if [ -z "${_SS_COPY:-}" ]; then
  _copy=$(mktemp /tmp/tmp_scratch_screen.XXXXXX.sh) && cp "${BASH_SOURCE[0]}" "$_copy" \
    && REPO_ROOT="$REPO_ROOT" _SS_COPY=1 exec bash "$_copy" "$@"
fi
trap 'rm -f "${BASH_SOURCE[0]}"' EXIT

SCREEN="${SCREEN:-scratch}"
ARMS="${ARMS:?set ARMS, e.g. ARMS=\"lr2e-4:LR=2e-4\" (see header)}"
read -ra BASE_ENV <<< "${BASE:-}"
HASH="${SCREEN_HASH:-$(git rev-parse --short HEAD)}"   # pin on relaunch
source scripts/require_clean_tree.sh "$HASH"           # committed code == pinned code
DRV_LOG="logs/v13_sempty_${SCREEN}_driver_${HASH}.log"
exec > >(tee -a "$DRV_LOG") 2>&1
echo "=== $SCREEN start $(date -u +%FT%TZ) SCREEN_HASH=$HASH arms='$ARMS' base='${BASE:-}' ==="

for spec in $ARMS; do
  arm="${spec%%:*}"; ovr=""; [ "$spec" != "$arm" ] && ovr="${spec#*:}"
  TAG="${SCREEN}_${arm}"
  CKPT_DIR="checkpoints_v13_sempty/${TAG}_${HASH}"
  if [ -f "$CKPT_DIR/.done" ]; then echo "=== arm $arm already done, skip ==="; continue; fi
  LOG=$(ls -t logs/v13_sempty_${TAG}_${HASH}_*.log 2>/dev/null | head -1)
  LOG="${LOG:-logs/v13_sempty_${TAG}_${HASH}_$(date +%Y%m%d_%H%M).log}"
  ENVS=()
  IFS=',' read -ra _kv <<< "$ovr"
  # A piece without "=" belongs to the previous value (SWA_LAYERS=5,9,13 stays whole).
  for kv in "${_kv[@]}"; do
    [ -z "$kv" ] && continue
    if [[ "$kv" != *=* && ${#ENVS[@]} -gt 0 ]]; then ENVS[-1]+=",$kv"; else ENVS+=("$kv"); fi
  done
  echo "=== arm $arm (${ENVS[*]:-no overrides}) ckpt=$CKPT_DIR log=$LOG $(date -u +%FT%TZ) ==="
  env TAG="$TAG" CKPT_DIR="$CKPT_DIR" RESUME_LOG="$LOG" PIN_HASH="$HASH" \
    SOURCES=dclm,fineweb,smoltalk2_mid,recall WEIGHTS=0.44,0.41,0.10,0.05 \
    CHRONO=1 OUT_GATE=1 DELTA=1 ANSWER_W=100 \
    TARGET_TOKENS=99532800 BLEND_WARMUP=0 WARMUP=100 VAL_EVERY=900 \
    SAVE_EVERY=900 KEEP_EVERY=0 GEN_EVERY=0 MAX_RETRIES=2 \
    "${BASE_ENV[@]}" "${ENVS[@]}" bash v13_sempty/tmp_pretrain_mix.sh
  RC=$?
  echo "=== arm $arm train exit=$RC $(date -u +%FT%TZ) ==="
  if [ "$RC" -eq 0 ]; then
    touch "$CKPT_DIR/.done"
    # latest.pt only resumes an unfinished arm; best_model.pt (+ any kept
    # milestone) carries the result. PRUNE_LATEST=0 keeps it.
    [ "${PRUNE_LATEST:-1}" = "1" ] && rm -f "$CKPT_DIR/latest.pt"
  else echo "=== arm $arm FAILED (diverged?) ==="; fi
done

for spec in $ARMS; do
  arm="${spec%%:*}"
  CK="checkpoints_v13_sempty/${SCREEN}_${arm}_${HASH}/best_model.pt"
  [ -f "checkpoints_v13_sempty/${SCREEN}_${arm}_${HASH}/.done" ] || { echo "skip probes: $arm not done"; continue; }
  PJ="logs/memory_probes/v13_sempty_${SCREEN}_${arm}_${HASH}_t40_behavioral.json"
  LJ="logs/pos_loss/v13_sempty_${SCREEN}_${arm}_${HASH}_pg19.json"
  [ -f "$PJ" ] || .venv/bin/python scripts/run_memory_behavioral.py --model-type v13_sempty \
    --checkpoint "$CK" --preset baseline_real_pm \
    --context-lengths 128,256,512,1024,2048,4096,8192 --positions 0,0.5,1 \
    --association-counts 1,4,8 --trials 40 --max-context 8192 \
    --output "$PJ" 2>&1 | grep -v scored
  [ -f "$LJ" ] || .venv/bin/python scripts/eval_pos_loss.py --model-type v13_sempty \
    --checkpoint "$CK" --context 8192 --docs 50 --output "$LJ"
done

echo "--- final holdout val_loss (last val of each arm) ---"
for spec in $ARMS; do
  arm="${spec%%:*}"
  log=$(ls -t logs/v13_sempty_${SCREEN}_${arm}_${HASH}_*.log 2>/dev/null | head -1)
  vl=$(grep -o "\[val @ step [0-9]*\] val_loss=[0-9.]*" "$log" 2>/dev/null | tail -1 | sed 's/.*=//')
  echo "  $arm: ${vl:-n/a}"
done
echo "=== $SCREEN DONE $(date -u +%FT%TZ) ==="
