#!/usr/bin/env bash
# v14 micro-bench launcher (B0 memory, B1 rules; v14/EXPERIMENTS_V14.md). Run in tmux:
#   tmux new-session -d -s v14b1 "TAG=b1 FAMILIES=rules ARMS=echo,op bash v14/run_bench.sh"
# Every arm trains on the same data; writes logs/v14_<TAG>_<hash>_<time>.{log,json}.
#
# Env: ARMS, FAMILIES (facts,bind,rebind | rules), TAG, STEPS, BATCH, AUX,
#   EXTRA (more microbench args, e.g. "--seed 7 --init_seed 1"),
#   SAVE=1 (keep each arm's weights in checkpoints_v14/<TAG>_<hash>_<time>/),
#   GPU, NEED_MB (free GPU memory to wait for), MEM_MAX (RAM cap, 0 = off),
#   WAIT_TMUX (tmux sessions that must end first),
#   WAIT_DONE ("file:pattern": the wait also ends when that log has the line),
#   RETRIES (default 2): if python dies by a signal (segfault, illegal
#   instruction), rerun it with --resume; arms already in the JSON are kept.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f scripts/require_clean_tree.sh ] || { echo "launch guard scripts/require_clean_tree.sh missing: REFUSE (restore: git show HEAD:scripts/require_clean_tree.sh > scripts/require_clean_tree.sh)"; exit 1; }   # fail closed (2026-11-16: the guard was found deleted)
source scripts/require_clean_tree.sh

ARMS="${ARMS:-echo,tpp}"
FAMILIES="${FAMILIES:-facts,bind,rebind}"
TAG="${TAG:-b0}"
STEPS="${STEPS:-6000}"
BATCH="${BATCH:-16}"
AUX="${AUX:-0}"
EXTRA="${EXTRA:-}"
SAVE="${SAVE:-0}"
NEED_MB="${NEED_MB:-14000}"
GPU="${GPU:-0}"
MEM_MAX="${MEM_MAX:-12G}"
# Free memory alone is not a safe gate (a user service stopping freed 8.8 GB
# and B0 started next to T1, 2026-10-29), and a tmux session can outlive its
# run (an `exec bash` tail held B1 back two hours, 2026-10-29): wait for the
# session to end OR for its log to print its DONE line.
WAIT_TMUX="${WAIT_TMUX:-}"
WAIT_DONE="${WAIT_DONE:-}"

for s in $WAIT_TMUX; do
  echo "v14 ${TAG}: waiting for tmux session '$s' to end ($(date))"
  while tmux has-session -t "$s" 2>/dev/null; do
    if [ -n "$WAIT_DONE" ] && grep -q -- "${WAIT_DONE#*:}" "${WAIT_DONE%%:*}" 2>/dev/null; then
      echo "v14 ${TAG}: '${WAIT_DONE#*:}' found in ${WAIT_DONE%%:*}, not waiting for '$s'"
      break
    fi
    sleep 60
  done
done

echo "v14 ${TAG}: waiting for >= ${NEED_MB} MB free on GPU ${GPU} ($(date))"
while :; do
  free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge "$NEED_MB" ] && break
  sleep 60
done

# Re-check after the wait: code committed while we waited must be clean too,
# or the hash in the log name would not describe the code that runs.
[ -f scripts/require_clean_tree.sh ] || { echo "launch guard scripts/require_clean_tree.sh missing: REFUSE (restore: git show HEAD:scripts/require_clean_tree.sh > scripts/require_clean_tree.sh)"; exit 1; }   # fail closed (2026-11-16: the guard was found deleted)
source scripts/require_clean_tree.sh
HASH=$(git rev-parse --short HEAD)
STAMP=$(date +%Y%m%d_%H%M)
LOG="logs/v14_${TAG}_${HASH}_${STAMP}.log"
JSON="logs/v14_${TAG}_${HASH}_${STAMP}.json"
SAVE_ARGS=()
if [ "$SAVE" = "1" ]; then
  SAVE_ARGS=(--save_dir "checkpoints_v14/${TAG}_${HASH}_${STAMP}")
fi
echo "v14 ${TAG} start $(date) commit=${HASH} families=${FAMILIES} arms=${ARMS} steps=${STEPS} batch=${BATCH} aux=${AUX} extra='${EXTRA}' save=${SAVE} log=${LOG}"
# RAM cap (user: do not flood RAM on the shared box): the scope is killed
# at MEM_MAX instead of hanging the machine.
CAP=()
if [ "$MEM_MAX" != "0" ] && command -v systemd-run >/dev/null; then
  CAP=(systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" --)
fi
# Python itself crashed twice on this box on 2026-10-30 (a segfault, an
# illegal instruction, both inside the interpreter binary), so a death by
# signal reruns with --resume. A Python error (exit < 128) does not: that is
# a bug, not a crash.
RETRIES="${RETRIES:-2}"
attempt=1
RESUME=()
while :; do
  set +e
  # shellcheck disable=SC2086  # EXTRA is a list of args
  CUDA_VISIBLE_DEVICES="$GPU" "${CAP[@]}" .venv/bin/python -u -m v14.microbench \
    --arms "$ARMS" --steps "$STEPS" --batch "$BATCH" --aux_weight "$AUX" --families "$FAMILIES" \
    --json "$JSON" "${SAVE_ARGS[@]}" "${RESUME[@]}" $EXTRA 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" -eq 0 ] && break
  if [ "$rc" -lt 128 ] || [ "$attempt" -gt "$RETRIES" ]; then
    echo "v14 ${TAG} FAILED rc=${rc} after ${attempt} attempt(s) $(date)" | tee -a "$LOG"
    exit "$rc"
  fi
  echo "v14 ${TAG}: python died by signal $((rc - 128)) (attempt ${attempt}); resuming from ${JSON} $(date)" | tee -a "$LOG"
  attempt=$((attempt + 1))
  RESUME=(--resume)
done
echo "v14 ${TAG} DONE $(date)"
