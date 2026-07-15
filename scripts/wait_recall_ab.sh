#!/usr/bin/env bash
# Wait until recall A/B finishes (all 5 verdicts) or the sweep process dies.
set -euo pipefail
cd "$(dirname "$0")/.."
ARMS=(control gate floor recall combo)
ROOT=checkpoints_v11_recall_ab
PIDFILE=logs/v11/recall_ab_sweep/LATEST_PID.txt
STATUS=logs/v11/recall_ab_sweep/monitor_status.txt
SUMMARY=logs/v11/recall_ab_sweep/final_summary.json

all_done() {
  local a
  for a in "${ARMS[@]}"; do
    [[ -f "$ROOT/$a/eval/verdict.json" ]] || return 1
  done
  return 0
}

n_done() {
  local n=0 a
  for a in "${ARMS[@]}"; do
    [[ -f "$ROOT/$a/eval/verdict.json" ]] && n=$((n+1))
  done
  echo "$n"
}

while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  pid=$(cat "$PIDFILE" 2>/dev/null || echo none)
  alive=0
  if [[ "$pid" != "none" ]] && kill -0 "$pid" 2>/dev/null; then alive=1; fi
  # also treat child train as alive
  if pgrep -f 'v11.train .*checkpoints_v11_recall_ab' >/dev/null; then alive=1; fi
  if pgrep -f 'run_v11_recall_ab' >/dev/null; then alive=1; fi

  nd=$(n_done)
  # latest log line
  latest=$(ls -t logs/v11/recall_ab_*/v11_*.log 2>/dev/null | head -1 || true)
  tail1=""
  if [[ -n "$latest" ]]; then
    tail1=$(tail -1 "$latest" | cut -c1-140)
  fi
  echo "[$ts] done=$nd/5 alive=$alive pid=$pid log=$latest | $tail1" | tee -a "$STATUS"

  if all_done; then
    echo "[$ts] ALL VERDICTS PRESENT" | tee -a "$STATUS"
    uv run python - <<'PY'
import json
from pathlib import Path
arms = ['control','gate','floor','recall','combo']
rows = []
for a in arms:
    v = json.loads(Path(f'checkpoints_v11_recall_ab/{a}/eval/verdict.json').read_text())
    b = v.get('behavioral', {})
    g = v.get('gate', {})
    c = v.get('criteria', {})
    rows.append({
        'arm': a,
        'ship': v.get('ship'),
        'single_at_max': b.get('single_assoc_at_max_context'),
        'single_by_ctx': b.get('single_assoc_by_context'),
        'overall_acc': b.get('overall_accuracy'),
        'p_cmf': g.get('mean_content_minus_filler'),
        'abs_cmf': g.get('abs_content_minus_filler'),
        'mean_gamma': g.get('mean_gamma'),
        'mean_protect': g.get('mean_protect'),
        'recall_pass': (c.get('recall_single_at_max_context') or {}).get('pass'),
        'gate_pass': (c.get('gate_selectivity_abs') or {}).get('pass'),
    })
# pull val_ppl from train logs if present
import re, glob
for r in rows:
    logs = sorted(glob.glob(f"logs/v11/recall_ab_{r['arm']}_*/v11_*.log"))
    ppls = []
    for lp in logs:
        for line in open(lp, errors='ignore'):
            m = re.search(r'Val Loss:.*PPL:\s*([0-9.]+)', line)
            if m: ppls.append(float(m.group(1)))
    r['val_ppl_last'] = ppls[-1] if ppls else None
rows_sorted = sorted(rows, key=lambda x: (-(x['single_at_max'] or -1), -(x['abs_cmf'] or -1)))
out = {'arms': rows, 'winner': rows_sorted[0]['arm'] if rows_sorted else None, 'ranked': [r['arm'] for r in rows_sorted]}
Path('logs/v11/recall_ab_sweep/final_summary.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
PY
    exit 0
  fi

  if [[ "$alive" -eq 0 && "$nd" -lt 5 ]]; then
    echo "[$ts] SWEEP DEAD with only $nd/5 verdicts" | tee -a "$STATUS"
    exit 2
  fi
  sleep 600
done
