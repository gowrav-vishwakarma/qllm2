#!/usr/bin/env bash
# Stage-3 matched behavioral baselines: V11 winner vs Mamba-130m HF.
# Transformer: optional TRANSFORMER_CHECKPOINT if trained separately.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_baselines

OUT_ROOT="${OUT_ROOT:-logs/v11/recall_baselines}"
V11_CKPT="${V11_CKPT:-}"
if [[ -z "$V11_CKPT" ]]; then
  if [[ -f checkpoints_v11_recall_fromscratch/best_run/eval/verdict.json ]]; then
    V11_CKPT=checkpoints_v11_recall_fromscratch/best_run/best_model.pt
  elif [[ -f checkpoints_v11_recall_ab/gate/eval/verdict.json ]]; then
    V11_CKPT=checkpoints_v11_recall_ab/gate/best_model.pt
  else
    V11_CKPT=checkpoints_v11_recall_ab/combo/best_model.pt
  fi
fi

BEHAVIOR_ARGS=(
  --context-lengths "${BEHAVIOR_CONTEXTS:-128,512,1024,2048}"
  --positions "${BEHAVIOR_POSITIONS:-0,0.5,1}"
  --association-counts "${BEHAVIOR_ASSOCIATIONS:-1,4,8}"
  --trials "${BEHAVIOR_TRIALS:-60}"
)

echo "[baselines] V11 checkpoint: $V11_CKPT" | tee "$OUT_ROOT/run.log"

uv run python scripts/run_memory_behavioral.py \
  --model-type v11 --checkpoint "$V11_CKPT" --preset v11_e3_k3_chat \
  --output "$OUT_ROOT/v11_behavior.json" \
  "${BEHAVIOR_ARGS[@]}" 2>&1 | tee -a "$OUT_ROOT/run.log"

uv run python scripts/run_memory_behavioral.py \
  --model-type hf --model-id "${MAMBA_MODEL:-state-spaces/mamba-130m-hf}" \
  --output "$OUT_ROOT/mamba_behavior.json" \
  "${BEHAVIOR_ARGS[@]}" 2>&1 | tee -a "$OUT_ROOT/run.log"

if [[ -n "${TRANSFORMER_CHECKPOINT:-}" && -f "$TRANSFORMER_CHECKPOINT" ]]; then
  uv run python scripts/run_memory_behavioral.py \
    --model-type transformer --checkpoint "$TRANSFORMER_CHECKPOINT" \
    --output "$OUT_ROOT/transformer_behavior.json" \
    "${BEHAVIOR_ARGS[@]}" 2>&1 | tee -a "$OUT_ROOT/run.log"
else
  echo "[baselines] skip transformer (set TRANSFORMER_CHECKPOINT to enable)" | tee -a "$OUT_ROOT/run.log"
fi

uv run python - <<'PY' | tee -a "$OUT_ROOT/run.log"
import json
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from behavioral_summary import behavioral_summary

def summary(path):
    d = json.loads(Path(path).read_text())
    b = behavioral_summary(d)
    return {
        'path': str(path),
        'params': d.get('parameter_count'),
        'single_at_max': b.get('single_assoc_at_max_context'),
        'singles': b.get('single_assoc_by_context'),
        'multi8_at_min': b.get('multi8_at_min_context'),
        'overall': b.get('overall_accuracy'),
    }

root = Path('logs/v11/recall_baselines')
rows = []
for name in ['v11_behavior.json', 'mamba_behavior.json', 'transformer_behavior.json',
             'mamba_matched_behavior.json', 'transformer_matched_behavior.json']:
    p = root / name
    if p.exists():
        rows.append({'model': name.replace('_behavior.json',''), **summary(p)})

out = {'comparisons': rows, 'metric_note':
       'single_at_max = mean over positions of associations==1 at max context '
       '(same as recall_gate_verdict.single_assoc_at_max_context)'}
Path('logs/v11/recall_baselines/summary.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
PY

echo "[baselines] done -> $OUT_ROOT/summary.json" | tee -a "$OUT_ROOT/run.log"
