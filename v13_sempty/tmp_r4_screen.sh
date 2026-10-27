#!/usr/bin/env bash
# R4 warm-start screen (EXPERIMENTS_SEMPY "R4"): two arms from the Phase 3b
# checkpoint, identical except --store, run SEQUENTIALLY (never concurrent),
# then the 40-trial recall probe + PG-19 per-position loss for both.
#
#   tmux new-session -d -s r4_screen "bash v13_sempty/tmp_r4_screen.sh"
#
# RESUMABLE / idempotent: re-run the SAME command with SCREEN_HASH=<hash of the
# first launch> (printed below + in SCRATCHPAD.md). A finished arm (.done
# marker) is skipped, a half-done arm resumes from its latest.pt (launcher
# --resume auto) appending to its existing log, finished probes are skipped.
#
# Env: SCREEN_TOKENS (default 300M), SCREEN_LR (5e-5: Phase 3b ended at LR~0,
# both arms re-warm the same way), INIT (source checkpoint), ARMS ("store ctrl").
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
if [ -z "${_R4_COPY:-}" ]; then
  _copy=$(mktemp /tmp/tmp_r4_screen.XXXXXX.sh) && cp "${BASH_SOURCE[0]}" "$_copy" \
    && REPO_ROOT="$REPO_ROOT" _R4_COPY=1 exec bash "$_copy" "$@"
fi
trap 'rm -f "${BASH_SOURCE[0]}"' EXIT

INIT="${INIT:-checkpoints_v13_sempty/mix3b_delta_answ100_1c913ef/best_model.pt}"
SCREEN_TOKENS="${SCREEN_TOKENS:-300000000}"
SCREEN_LR="${SCREEN_LR:-5e-5}"
ARMS="${ARMS:-store ctrl}"
HASH="${SCREEN_HASH:-$(git rev-parse --short HEAD)}"   # pin on relaunch
echo "=== r4 screen start $(date -u +%FT%TZ) SCREEN_HASH=$HASH arms='$ARMS' ==="

for arm in $ARMS; do
  TAG="r4screen_${arm}"
  CKPT_DIR="checkpoints_v13_sempty/${TAG}_${HASH}"
  if [ -f "$CKPT_DIR/.done" ]; then echo "=== arm $arm already done, skip ==="; continue; fi
  OLD_LOG=$(ls -t logs/v13_sempty_${TAG}_${HASH}_*.log 2>/dev/null | head -1)
  STORE=0; [ "$arm" = "store" ] && STORE=1
  echo "=== arm $arm (STORE=$STORE) ckpt=$CKPT_DIR log=${OLD_LOG:-new} $(date -u +%FT%TZ) ==="
  TAG="$TAG" CKPT_DIR="$CKPT_DIR" RESUME_LOG="$OLD_LOG" STORE="$STORE" INIT_FROM="$INIT" \
  SOURCES=dclm,fineweb,smoltalk2_mid,recall WEIGHTS=0.44,0.41,0.10,0.05 \
  DELTA=1 ANSWER_W=100 TARGET_TOKENS="$SCREEN_TOKENS" BLEND_WARMUP=0 \
  LR="$SCREEN_LR" WARMUP=200 VAL_EVERY=1000 SAVE_EVERY=1000 KEEP_EVERY=0 \
  GEN_EVERY=4000 MAX_RETRIES=3 \
    bash v13_sempty/tmp_pretrain_mix.sh
  RC=$?
  echo "=== arm $arm train exit=$RC $(date -u +%FT%TZ) ==="
  if [ "$RC" -eq 0 ]; then touch "$CKPT_DIR/.done"; else echo "arm $arm failed; stop"; exit "$RC"; fi
done

for arm in $ARMS; do
  CK="checkpoints_v13_sempty/r4screen_${arm}_${HASH}/best_model.pt"
  [ -f "$CK" ] || { echo "missing $CK"; continue; }
  PJ="logs/memory_probes/v13_sempty_r4screen_${arm}_${HASH}_t40_behavioral.json"
  LJ="logs/pos_loss/v13_sempty_r4screen_${arm}_${HASH}_pg19.json"
  [ -f "$PJ" ] || .venv/bin/python scripts/run_memory_behavioral.py --model-type v13_sempty \
    --checkpoint "$CK" --preset baseline_real_pm \
    --context-lengths 128,256,512,1024,2048,4096,8192 --positions 0,0.5,1 \
    --association-counts 1,4,8 --trials 40 --max-context 8192 \
    --output "$PJ" 2>&1 | grep -v scored
  [ -f "$LJ" ] || .venv/bin/python scripts/eval_pos_loss.py --model-type v13_sempty \
    --checkpoint "$CK" --context 8192 --docs 50 --output "$LJ"
done
echo "=== r4 screen DONE $(date -u +%FT%TZ) ==="
