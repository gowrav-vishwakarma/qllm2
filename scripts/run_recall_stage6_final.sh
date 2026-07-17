#!/usr/bin/env bash
# Stage-6d final: compare winner vs matched baselines; append EXPERIMENTS_V11.md.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_stage6_final

export BEHAVIOR_TRIALS="${BEHAVIOR_TRIALS:-60}"
export FINEWEB_LOCAL_DIR="${FINEWEB_LOCAL_DIR:-data/fineweb-edu/sample-10BT}"
export HF_HUB_DISABLE_XET=1

WIN_CKPT="${WIN_CKPT:-checkpoints_v11_recall_stage6_combo/best_run/best_model.pt}"
if [[ ! -f "$WIN_CKPT" ]]; then
  # Fall back to best individual stage6 arm
  WIN_CKPT="$(uv run python - <<'PY'
from pathlib import Path
import json
s=Path('logs/v11/recall_stage6/summary.json')
if s.exists():
    d=json.loads(s.read_text())
    w=d.get('winner')
    if w:
        p=Path(f'checkpoints_v11_recall_stage6/{w}/best_model.pt')
        if p.exists():
            print(p); raise SystemExit
print('')
PY
)"
fi

OUT="logs/v11/recall_stage6_final"
mkdir -p "$OUT"

if [[ -n "$WIN_CKPT" && -f "$WIN_CKPT" ]]; then
  echo "[final] winner ckpt=$WIN_CKPT"
  if [[ ! -f "$OUT/winner_behavior.json" ]]; then
    uv run python scripts/run_memory_behavioral.py \
      --model-type v11 --checkpoint "$WIN_CKPT" --preset v11_e3_k3_chat \
      --output "$OUT/winner_behavior.json" \
      --context-lengths 128,512,1024,2048 --positions 0,0.5,1 \
      --association-counts 1,4,8 --trials "$BEHAVIOR_TRIALS" || true
  fi
else
  echo "[final] WARN: no winner checkpoint"
fi

# Ensure matched baselines exist in summary (re-summarize)
uv run python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from behavioral_summary import behavioral_summary

def load_row(path, model):
    d = json.loads(Path(path).read_text())
    b = behavioral_summary(d)
    return {
        'model': model,
        'path': str(path),
        'params': d.get('parameter_count'),
        'single_at_max': b.get('single_assoc_at_max_context'),
        'singles': b.get('single_assoc_by_context'),
        'multi8_at_min': b.get('multi8_at_min_context'),
        'overall': b.get('overall_accuracy'),
    }

rows = []
pairs = [
    ('logs/v11/recall_stage6_final/winner_behavior.json', 'v11_stage6_winner'),
    ('logs/v11/recall_baselines/v11_behavior.json', 'v11_stage4_fromscratch'),
    ('logs/v11/recall_baselines/mamba_matched_behavior.json', 'mamba_matched'),
    ('logs/v11/recall_baselines/transformer_matched_behavior.json', 'transformer_matched'),
    ('logs/v11/recall_baselines/mamba_behavior.json', 'mamba_pretrained_hf'),
]
for path, name in pairs:
    if Path(path).exists():
        rows.append(load_row(path, name))

# Micro + stage6 arm tables
micro = Path('logs/v11/recall_micro/summary.json')
stage6 = Path('logs/v11/recall_stage6/summary.json')
combo_v = Path('checkpoints_v11_recall_stage6_combo/best_run/eval/verdict.json')
out = {
    'comparisons': rows,
    'micro': json.loads(micro.read_text()) if micro.exists() else None,
    'stage6_arms': json.loads(stage6.read_text()) if stage6.exists() else None,
    'combo_verdict': json.loads(combo_v.read_text()) if combo_v.exists() else None,
}
Path('logs/v11/recall_stage6_final/summary.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
PY

# Append documentation to EXPERIMENTS_V11.md if not already present
uv run python scripts/append_stage6_experiments.py

echo "[final] done -> logs/v11/recall_stage6_final/summary.json"
