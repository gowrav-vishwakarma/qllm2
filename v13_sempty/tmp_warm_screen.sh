#!/usr/bin/env bash
# Warm-start screen: N arms continued from one checkpoint (default Phase 3b),
# identical except each arm's env overrides, run SEQUENTIALLY (never
# concurrent), then the 40-trial recall probe + PG-19 per-position loss for
# every arm. First used for R4 (EXPERIMENTS_SEMPY "R4", @0868a1b).
#
#   SCREEN=r5screen ARMS="new:FOO=1,BAR=2" \
#     tmux new-session -d -s r5_screen "bash v13_sempty/tmp_warm_screen.sh"
#
# ARMS: space-separated "name[:VAR=val,VAR=val]"; each VAR is passed to
# tmp_pretrain_mix.sh for that arm only ("ctrl" = no overrides). Reusable
# control: the R4 ctrl arm (checkpoints_v13_sempty/r4screen_ctrl_0868a1b,
# probe/pos-loss jsons in logs/) IS this recipe's control (same init, tokens,
# LR, data order), so a new screen can run only its new arm and compare to it.
#
# RESUMABLE / idempotent: re-run the SAME command with SCREEN_HASH=<hash of the
# first launch> (printed below + in SCRATCHPAD.md). A finished arm (.done
# marker) is skipped, a half-done arm resumes from its latest.pt (launcher
# --resume auto) appending to its existing log, finished probes are skipped.
#
# Env: SCREEN (tag prefix, default warmscreen), SCREEN_TOKENS (default 300M),
# SCREEN_LR (5e-5: Phase 3b ended at LR~0, every arm re-warms the same way),
# INIT (source checkpoint).
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
if [ -z "${_WS_COPY:-}" ]; then
  _copy=$(mktemp /tmp/tmp_warm_screen.XXXXXX.sh) && cp "${BASH_SOURCE[0]}" "$_copy" \
    && REPO_ROOT="$REPO_ROOT" _WS_COPY=1 exec bash "$_copy" "$@"
fi
trap 'rm -f "${BASH_SOURCE[0]}"' EXIT

SCREEN="${SCREEN:-warmscreen}"
INIT="${INIT:-checkpoints_v13_sempty/mix3b_delta_answ100_1c913ef/best_model.pt}"
SCREEN_TOKENS="${SCREEN_TOKENS:-300000000}"
SCREEN_LR="${SCREEN_LR:-5e-5}"
ARMS="${ARMS:?set ARMS, e.g. ARMS=\"new:FOO=1\" (see header)}"
HASH="${SCREEN_HASH:-$(git rev-parse --short HEAD)}"   # pin on relaunch
source scripts/require_clean_tree.sh "$HASH"           # committed code == pinned code
DRV_LOG="logs/v13_sempty_${SCREEN}_driver_${HASH}.log"
exec > >(tee -a "$DRV_LOG") 2>&1
echo "=== $SCREEN start $(date -u +%FT%TZ) SCREEN_HASH=$HASH arms='$ARMS' init=$INIT ==="

for spec in $ARMS; do
  arm="${spec%%:*}"; ovr=""; [ "$spec" != "$arm" ] && ovr="${spec#*:}"
  TAG="${SCREEN}_${arm}"
  CKPT_DIR="checkpoints_v13_sempty/${TAG}_${HASH}"
  if [ -f "$CKPT_DIR/.done" ]; then echo "=== arm $arm already done, skip ==="; continue; fi
  OLD_LOG=$(ls -t logs/v13_sempty_${TAG}_${HASH}_*.log 2>/dev/null | head -1)
  # Name a new arm log after the PINNED hash (not HEAD), so a docs-only pull
  # mid-screen cannot rename it.
  OLD_LOG="${OLD_LOG:-logs/v13_sempty_${TAG}_${HASH}_$(date +%Y%m%d_%H%M).log}"
  ENVS=()
  IFS=',' read -ra _kv <<< "$ovr"
  for kv in "${_kv[@]}"; do [ -n "$kv" ] && ENVS+=("$kv"); done
  echo "=== arm $arm (${ENVS[*]:-no overrides}) ckpt=$CKPT_DIR log=$OLD_LOG $(date -u +%FT%TZ) ==="
  env TAG="$TAG" CKPT_DIR="$CKPT_DIR" RESUME_LOG="$OLD_LOG" PIN_HASH="$HASH" INIT_FROM="$INIT" \
    SOURCES=dclm,fineweb,smoltalk2_mid,recall WEIGHTS=0.44,0.41,0.10,0.05 \
    DELTA=1 ANSWER_W=100 TARGET_TOKENS="$SCREEN_TOKENS" BLEND_WARMUP=0 \
    LR="$SCREEN_LR" WARMUP=200 VAL_EVERY=1000 SAVE_EVERY=1000 KEEP_EVERY=0 \
    GEN_EVERY=4000 MAX_RETRIES=3 "${ENVS[@]}" \
    bash v13_sempty/tmp_pretrain_mix.sh
  RC=$?
  echo "=== arm $arm train exit=$RC $(date -u +%FT%TZ) ==="
  if [ "$RC" -eq 0 ]; then touch "$CKPT_DIR/.done"; else echo "arm $arm failed; stop"; exit "$RC"; fi
done

for spec in $ARMS; do
  arm="${spec%%:*}"
  CK="checkpoints_v13_sempty/${SCREEN}_${arm}_${HASH}/best_model.pt"
  [ -f "$CK" ] || { echo "missing $CK"; continue; }
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
echo "=== $SCREEN DONE $(date -u +%FT%TZ) ==="
