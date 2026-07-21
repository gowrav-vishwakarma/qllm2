#!/usr/bin/env bash
# Stage-6a: train matched ~100M Transformer + Mamba on fineweb+recall (1B tok each),
# then run behavioral suite. Fair architecture comparison (not pretrained HF Mamba).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_matched_baselines \
         checkpoints_v11_recall_matched/transformer \
         checkpoints_v11_recall_matched/mamba

export FINEWEB_LOCAL_DIR="${FINEWEB_LOCAL_DIR:-data/fineweb-edu/sample-10BT}"
export HF_HUB_DISABLE_XET=1
export BEHAVIOR_TRIALS="${BEHAVIOR_TRIALS:-60}"

TOKEN_BUDGET="${TOKEN_BUDGET:-1000000000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RECALL_WEIGHT="${RECALL_WEIGHT:-3}"
WEB_WEIGHT=$((96 - RECALL_WEIGHT))
OUT_T="checkpoints_v11_recall_matched/transformer"
OUT_M="checkpoints_v11_recall_matched/mamba"
LOG_ROOT="logs/v11/recall_matched_baselines"
EVAL_ROOT="logs/v11/recall_baselines"

# ── Transformer ──────────────────────────────────────────────────────────────
if [[ ! -f "$OUT_T/best_model.pt" ]]; then
  echo "[matched] train transformer budget=$TOKEN_BUDGET" | tee -a "$LOG_ROOT/master.log"
  (
    set +e
    uv run python scripts/train_matched_baseline.py \
      --arch transformer --token_budget "$TOKEN_BUDGET" --batch_size "$BATCH_SIZE" \
      --pretrain_sources fineweb,recall --pretrain_weights "${WEB_WEIGHT},${RECALL_WEIGHT}" \
      --checkpoint_dir "$OUT_T" --save_every_steps 2000
    exit 0
  ) 2>&1 | tee -a "$LOG_ROOT/transformer.log"
else
  echo "[matched] skip transformer train (ckpt exists)" | tee -a "$LOG_ROOT/master.log"
fi

# ── Mamba ────────────────────────────────────────────────────────────────────
# Sequential HF Mamba (no mamba-ssm / causal-conv1d) is ~15 tok/s here → 1B tok
# would take months. Skip unless kernels are installed or FORCE_SEQUENTIAL_MAMBA=1.
_has_mamba_kernels="$(uv run python - <<'PY'
import importlib.util as u
print('1' if u.find_spec('mamba_ssm') and u.find_spec('causal_conv1d') else '0')
PY
)"
if [[ -f "$OUT_M/best_hf/config.json" ]]; then
  echo "[matched] skip mamba train (ckpt exists)" | tee -a "$LOG_ROOT/master.log"
elif [[ "${SKIP_MATCHED_MAMBA:-0}" == "1" || ( "$_has_mamba_kernels" != "1" && "${FORCE_SEQUENTIAL_MAMBA:-0}" != "1" ) ]]; then
  mkdir -p "$OUT_M"
  echo "{\"skipped\": true, \"reason\": \"no mamba-ssm/causal-conv1d; sequential ~15 tok/s infeasible for ${TOKEN_BUDGET} tok\", \"has_kernels\": ${_has_mamba_kernels}}" \
    | tee "$OUT_M/SKIPPED.json" | tee -a "$LOG_ROOT/master.log"
  echo "[matched] SKIP mamba train (install mamba-ssm+causal-conv1d, or set FORCE_SEQUENTIAL_MAMBA=1)" | tee -a "$LOG_ROOT/master.log"
else
  echo "[matched] train mamba budget=$TOKEN_BUDGET kernels=$_has_mamba_kernels" | tee -a "$LOG_ROOT/master.log"
  (
    set +e
    uv run python scripts/train_matched_baseline.py \
      --arch mamba --size 100m --token_budget "$TOKEN_BUDGET" --batch_size 4 \
      --pretrain_sources fineweb,recall --pretrain_weights "${WEB_WEIGHT},${RECALL_WEIGHT}" \
      --checkpoint_dir "$OUT_M" --save_every_steps 2000
    exit 0
  ) 2>&1 | tee -a "$LOG_ROOT/mamba.log"
fi

# ── Behavioral eval ──────────────────────────────────────────────────────────
mkdir -p "$EVAL_ROOT"
BEHAVIOR_ARGS=(
  --context-lengths "${BEHAVIOR_CONTEXTS:-128,512,1024,2048}"
  --positions "${BEHAVIOR_POSITIONS:-0,0.5,1}"
  --association-counts "${BEHAVIOR_ASSOCIATIONS:-1,4,8}"
  --trials "${BEHAVIOR_TRIALS}"
)

if [[ -f "$OUT_T/best_model.pt" ]]; then
  echo "[matched] eval transformer" | tee -a "$LOG_ROOT/master.log"
  uv run python scripts/run_memory_behavioral.py \
    --model-type transformer --checkpoint "$OUT_T/best_model.pt" \
    --output "$EVAL_ROOT/transformer_matched_behavior.json" \
    "${BEHAVIOR_ARGS[@]}" 2>&1 | tee -a "$LOG_ROOT/eval.log" || true
fi

if [[ -f "$OUT_M/best_hf/config.json" ]]; then
  echo "[matched] eval mamba" | tee -a "$LOG_ROOT/master.log"
  # Prefer local tokenizer if saved; else GPT-2 (matched vocab)
  uv run python scripts/run_memory_behavioral.py \
    --model-type hf --model-id "$OUT_M/best_hf" \
    --output "$EVAL_ROOT/mamba_matched_behavior.json" \
    "${BEHAVIOR_ARGS[@]}" 2>&1 | tee -a "$LOG_ROOT/eval.log" || true
fi

# Refresh summary.json (includes matched + any prior rows)
TRANSFORMER_CHECKPOINT="$OUT_T/best_model.pt" \
  V11_CKPT="${V11_CKPT:-checkpoints_v11_recall_fromscratch/best_run/best_model.pt}" \
  BEHAVIOR_TRIALS="$BEHAVIOR_TRIALS" \
  bash -c '
    # Only re-summarize if behavior JSONs exist; avoid re-running full V11/HF evals
    uv run python - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from behavioral_summary import behavioral_summary
root = Path("logs/v11/recall_baselines")
rows = []
for name in ["v11_behavior.json", "mamba_behavior.json", "transformer_behavior.json",
             "mamba_matched_behavior.json", "transformer_matched_behavior.json"]:
    p = root / name
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    b = behavioral_summary(d)
    rows.append({
        "model": name.replace("_behavior.json", ""),
        "path": str(p),
        "params": d.get("parameter_count"),
        "single_at_max": b.get("single_assoc_at_max_context"),
        "singles": b.get("single_assoc_by_context"),
        "multi8_at_min": b.get("multi8_at_min_context"),
        "overall": b.get("overall_accuracy"),
    })
out = {"comparisons": rows,
       "metric_note": "single_at_max = mean over positions of associations==1 at max context"}
(root / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
  '

echo "[matched] done"
