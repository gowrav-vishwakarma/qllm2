#!/usr/bin/env bash
# Master orchestrator: download data -> hypersweep -> from-scratch -> baselines.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_pipeline

LOG="logs/v11/recall_pipeline/master.log"
: > "$LOG"

echo "[pipeline] $(date -u +%FT%TZ) start" | tee -a "$LOG"

echo "[pipeline] step 1: download FineWeb shards" | tee -a "$LOG"
bash scripts/download_fineweb_shards.sh 2>&1 | tee -a "$LOG"

echo "[pipeline] step 2: hyper sweeps (150M/arm)" | tee -a "$LOG"
bash scripts/run_recall_hypersweep.sh 2>&1 | tee -a "$LOG"

echo "[pipeline] step 3: from-scratch + routing levers (1B tok)" | tee -a "$LOG"
bash scripts/run_recall_fromscratch.sh 2>&1 | tee -a "$LOG"

echo "[pipeline] step 4: matched baselines" | tee -a "$LOG"
bash scripts/run_recall_baselines.sh 2>&1 | tee -a "$LOG"

echo "[pipeline] $(date -u +%FT%TZ) complete" | tee -a "$LOG"
