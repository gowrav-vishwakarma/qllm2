#!/usr/bin/env bash
# Matched contrastive recall: same seeds, lengths, trials for V13 / Transformer / Mamba.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
OUT="checkpoints_v13/matched_recall"
mkdir -p "$OUT"

COMMON=(
  --context-lengths 128,512,2048
  --positions 0,0.5,1
  --association-counts 1
  --trials 60
  --seed 1000
  --candidate-count 8
)

echo "=== V13 PAM (latest.pt, 4.1M tokens, train seq 2048) ==="
"$PYTHON" scripts/run_memory_behavioral.py \
  --model-type v13 \
  --checkpoint checkpoints_v13/smoke_recall_v2/latest.pt \
  --preset v13_micro_10m_recall \
  --output "$OUT/v13.json" \
  "${COMMON[@]}"

echo "=== Transformer (final_model.pt, 3.0M tokens, train seq 512) ==="
"$PYTHON" scripts/run_memory_behavioral.py \
  --model-type transformer \
  --checkpoint checkpoints_v13/transformer_micro_recall/final_model.pt \
  --output "$OUT/transformer.json" \
  "${COMMON[@]}"

echo "=== Mamba (best_hf, 2.0M tokens, train seq 512) ==="
"$PYTHON" scripts/run_memory_behavioral.py \
  --model-type hf \
  --model-id checkpoints_v13/mamba_micro_recall/best_hf \
  --output "$OUT/mamba.json" \
  "${COMMON[@]}"

"$PYTHON" - <<'PY'
import json, math
from pathlib import Path

def wilson(k, n, z=1.96):
    if n <= 0:
        return (float('nan'), float('nan'), float('nan'))
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / den
    return p, max(0.0, centre-half), min(1.0, centre+half)

out = Path('checkpoints_v13/matched_recall')
print('\nMatched recall (8-way contrastive, chance=12.5%, 60 trials/cell)\n')
print(f"{'model':<14} {'ctx':>5} {'acc':>7} {'95% CI':>18} {'n':>4}  notes")
print('-'*64)
for name in ('v13', 'transformer', 'mamba'):
    d = json.loads((out / f'{name}.json').read_text())
    skipped = sum(1 for r in d['rows'] if r.get('skipped'))
    for a in d['aggregates']:
        if a['associations'] != 1:
            continue
        n = a['n']
        acc = a['accuracy']
        k = int(round(acc * n))
        _, lo, hi = wilson(k, n)
        note = f"skip={skipped}" if skipped and a['context_tokens']==128 else ''
        print(f"{name:<14} {a['context_tokens']:>5} {acc:>6.1%}  [{lo:5.1%},{hi:5.1%}] {n:>4}  pos={a['target_position']}")
PY
echo "Done. JSON in checkpoints_v13/matched_recall/"
