#!/usr/bin/env bash
# B0 + N1 micro-bench (v14/EXPERIMENTS_V14.md "B0", "N1"). Run in tmux:
#   tmux new-session -d -s v14b0 "bash v14/run_b0.sh"
# Waits until the GPU has NEED_MB free (another run may hold it), then trains
# every arm on the same data and writes logs/v14_b0_<hash>_<time>.{log,json}.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/require_clean_tree.sh

ARMS="${ARMS:-delta,echo,n1sum,n1gate,n1phase}"
FAMILIES="${FAMILIES:-facts,bind,rebind}"   # B1: FAMILIES=rules TAG=b1
TAG="${TAG:-b0}"
STEPS="${STEPS:-6000}"
BATCH="${BATCH:-16}"
AUX="${AUX:-0}"
NEED_MB="${NEED_MB:-14000}"
GPU="${GPU:-0}"
# WAIT_TMUX: space-separated tmux sessions that must be gone before start
# (another session's training run). Free memory alone is not a safe gate:
# a user service stopping freed 8.8 GB and B0 started next to T1 (2026-10-29).
WAIT_TMUX="${WAIT_TMUX:-}"

for s in $WAIT_TMUX; do
  echo "v14 B0: waiting for tmux session '$s' to end ($(date))"
  while tmux has-session -t "$s" 2>/dev/null; do sleep 60; done
done

echo "v14 B0: waiting for >= ${NEED_MB} MB free on GPU ${GPU} ($(date))"
while :; do
  free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge "$NEED_MB" ] && break
  sleep 60
done

# Re-check after the wait: code committed while we waited must be clean too,
# or the hash in the log name would not describe the code that runs.
source scripts/require_clean_tree.sh
HASH=$(git rev-parse --short HEAD)
STAMP=$(date +%Y%m%d_%H%M)
LOG="logs/v14_${TAG}_${HASH}_${STAMP}.log"
JSON="logs/v14_${TAG}_${HASH}_${STAMP}.json"
echo "v14 ${TAG} start $(date) commit=${HASH} families=${FAMILIES} arms=${ARMS} steps=${STEPS} batch=${BATCH} aux=${AUX} log=${LOG}"
# RAM cap (user: do not flood RAM on the shared box): the scope is killed
# at MEM_MAX instead of hanging the machine. MEM_MAX=0 disables it.
MEM_MAX="${MEM_MAX:-12G}"
CAP=()
if [ "$MEM_MAX" != "0" ] && command -v systemd-run >/dev/null; then
  CAP=(systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" --)
fi
CUDA_VISIBLE_DEVICES="$GPU" "${CAP[@]}" .venv/bin/python -u -m v14.microbench \
  --arms "$ARMS" --steps "$STEPS" --batch "$BATCH" --aux_weight "$AUX" --families "$FAMILIES" --json "$JSON" 2>&1 | tee "$LOG"
echo "v14 ${TAG} DONE $(date)"
