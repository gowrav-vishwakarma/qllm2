#!/usr/bin/env bash
# B0 + N1 micro-bench (v14/EXPERIMENTS_V14.md "B0", "N1"). Run in tmux:
#   tmux new-session -d -s v14b0 "bash v14/run_b0.sh"
# Waits until the GPU has NEED_MB free (another run may hold it), then trains
# every arm on the same data and writes logs/v14_b0_<hash>_<time>.{log,json}.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/require_clean_tree.sh

ARMS="${ARMS:-delta,echo,n1sum,n1gate,n1phase}"
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

HASH=$(git rev-parse --short HEAD)
STAMP=$(date +%Y%m%d_%H%M)
LOG="logs/v14_b0_${HASH}_${STAMP}.log"
JSON="logs/v14_b0_${HASH}_${STAMP}.json"
echo "v14 B0 start $(date) commit=${HASH} arms=${ARMS} steps=${STEPS} batch=${BATCH} aux=${AUX} log=${LOG}"
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -u -m v14.microbench \
  --arms "$ARMS" --steps "$STEPS" --batch "$BATCH" --aux_weight "$AUX" --json "$JSON" 2>&1 | tee "$LOG"
echo "v14 B0 DONE $(date)"
