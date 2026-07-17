#!/usr/bin/env bash
# Stage-6 master orchestrator (unattended).
# Order: micro-capacity → architecture arms → combo → matched baselines → final compare.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_stage6_pipeline

export FINEWEB_LOCAL_DIR="${FINEWEB_LOCAL_DIR:-data/fineweb-edu/sample-10BT}"
export HF_HUB_DISABLE_XET=1
export BEHAVIOR_TRIALS="${BEHAVIOR_TRIALS:-60}"

LOG="logs/v11/recall_stage6_pipeline/nohup.log"
STATUS="logs/v11/recall_stage6_pipeline/status.log"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG" "$STATUS"; }

log "STAGE6 PIPELINE START"

# ── 6b micro-capacity ────────────────────────────────────────────────────────
if [[ -f logs/v11/recall_micro/summary.json ]]; then
  log "SKIP micro (summary exists)"
else
  log "START micro-capacity"
  bash scripts/run_recall_micro_capacity.sh 2>&1 | tee -a "$LOG" || log "WARN micro had errors"
  log "DONE micro-capacity"
fi

# ── 6c architecture arms ─────────────────────────────────────────────────────
if [[ -f logs/v11/recall_stage6/summary.json ]]; then
  # Re-run only if incomplete arms remain
  pending=$(uv run python - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('logs/v11/recall_stage6/summary.json').read_text())
print(sum(1 for r in s.get('arms',[]) if r.get('recall_at_2048') is None or r.get('status')=='pending'))
PY
)
  if [[ "$pending" == "0" ]]; then
    log "SKIP stage6 arms (complete)"
  else
    log "RESUME stage6 arms ($pending pending)"
    bash scripts/run_recall_stage6_arms.sh 2>&1 | tee -a "$LOG" || log "WARN arms had errors"
  fi
else
  log "START stage6 arms"
  bash scripts/run_recall_stage6_arms.sh 2>&1 | tee -a "$LOG" || log "WARN arms had errors"
  log "DONE stage6 arms"
fi

# ── 6d combo confirmation ────────────────────────────────────────────────────
if [[ -f checkpoints_v11_recall_stage6_combo/best_run/eval/verdict.json ]]; then
  log "SKIP combo (verdict exists)"
else
  log "START combo"
  bash scripts/run_recall_stage6_combo.sh 2>&1 | tee -a "$LOG" || log "WARN combo had errors"
  log "DONE combo"
fi

# ── matched baselines (fair Mamba/Transformer) ───────────────────────────────
if [[ -f logs/v11/recall_baselines/mamba_matched_behavior.json \
   && -f logs/v11/recall_baselines/transformer_matched_behavior.json ]]; then
  log "SKIP matched baselines (behavior JSONs exist)"
else
  log "START matched baselines"
  bash scripts/run_recall_matched_baselines.sh 2>&1 | tee -a "$LOG" || log "WARN matched had errors"
  log "DONE matched baselines"
fi

# ── final compare + document ─────────────────────────────────────────────────
log "START final compare"
bash scripts/run_recall_stage6_final.sh 2>&1 | tee -a "$LOG" || log "WARN final had errors"
log "STAGE6 PIPELINE COMPLETE"
