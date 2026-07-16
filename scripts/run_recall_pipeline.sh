#!/usr/bin/env bash
# Resumable master orchestrator: hypersweep -> from-scratch -> baselines.
# Skips download if shards exist; skips completed pipeline steps.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_pipeline

LOG="logs/v11/recall_pipeline/master.log"

log() { echo "[pipeline] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

SHARD_COUNT=$(ls data/fineweb-edu/sample-10BT/*.parquet 2>/dev/null | wc -l)
if [[ "$SHARD_COUNT" -lt 5 ]]; then
  log "step 1: download FineWeb shards (have $SHARD_COUNT)"
  bash scripts/download_fineweb_shards.sh 2>&1 | tee -a "$LOG" || log "download had errors, continuing"
else
  log "step 1: SKIP download ($SHARD_COUNT shards present)"
fi

if [[ ! -f logs/v11/recall_hypersweep/best_hypers.json ]]; then
  log "step 2: hyper sweeps (150M/arm)"
  bash scripts/run_recall_hypersweep.sh 2>&1 | tee -a "$LOG" || log "hypersweep had errors, continuing"
else
  log "step 2: SKIP hypersweep (best_hypers.json exists)"
fi

if [[ ! -f checkpoints_v11_recall_fromscratch/best_run/eval/verdict.json ]]; then
  log "step 3: from-scratch + routing levers (1B tok)"
  bash scripts/run_recall_fromscratch.sh 2>&1 | tee -a "$LOG" || log "from-scratch had errors, continuing"
else
  log "step 3: SKIP from-scratch (verdict exists)"
fi

if [[ ! -f logs/v11/recall_baselines/summary.json ]]; then
  log "step 4: matched baselines"
  bash scripts/run_recall_baselines.sh 2>&1 | tee -a "$LOG" || log "baselines had errors"
else
  log "step 4: SKIP baselines (summary exists)"
fi

log "complete"
