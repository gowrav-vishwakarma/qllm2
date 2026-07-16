#!/usr/bin/env bash
# Print recall pipeline progress snapshot.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== $(date -u +%FT%TZ) ==="
echo "GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo n/a)"
pgrep -af 'v11.train|run_recall' | grep -v pgrep | head -3 || echo "no active train"

echo ""
echo "--- Stage-2 A/B (done) ---"
python3 - <<'PY' 2>/dev/null || true
import json
d=json.load(open('logs/v11/recall_ab_sweep/final_summary.json'))
for a in d['arms']:
    print(f"  {a['arm']:8s} @2048={a['single_at_max']:.3f} ppl={a['val_ppl_last']}")
PY

echo ""
echo "--- Hypersweep ($(ls checkpoints_v11_recall_hypersweep/*/eval/verdict.json 2>/dev/null | wc -l)/13) ---"
for d in checkpoints_v11_recall_hypersweep/*/; do
  arm=$(basename "$d")
  if [[ -f "$d/eval/verdict.json" ]]; then
    python3 -c "import json; v=json.load(open('$d/eval/verdict.json')); b=v['behavioral']; print(f'  {arm}: @2048={b[\"single_assoc_at_max_context\"]:.3f} DONE')" 2>/dev/null
  elif [[ -f "$d/best_model.pt" ]]; then
    echo "  $arm: trained, eval pending"
  else
    echo "  $arm: pending"
  fi
done

echo ""
echo "--- From-scratch ---"
[[ -f checkpoints_v11_recall_fromscratch/best_run/eval/verdict.json ]] && echo "  DONE" || echo "  pending"

echo ""
echo "--- Baselines ---"
[[ -f logs/v11/recall_baselines/summary.json ]] && cat logs/v11/recall_baselines/summary.json | head -5 || echo "  pending"

echo ""
latest=$(ls -t logs/v11/recall_hypersweep/*/v11_*.log 2>/dev/null | head -1)
if [[ -n "$latest" ]]; then
  echo "Latest train: $latest"
  grep -E 'gtok=|Token budget|complete' "$latest" 2>/dev/null | tail -2
fi
