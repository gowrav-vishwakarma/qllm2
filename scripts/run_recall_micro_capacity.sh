#!/usr/bin/env bash
# Stage-6b: capacity micro-tests (~11M PAM variants + tiny Mamba) on 100% recall curriculum.
# Isolates storage capacity from language modeling. ~30M tokens/arm.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_micro checkpoints_v11_recall_micro

export FINEWEB_LOCAL_DIR="${FINEWEB_LOCAL_DIR:-data/fineweb-edu/sample-10BT}"
export HF_HUB_DISABLE_XET=1
export BEHAVIOR_TRIALS="${BEHAVIOR_TRIALS:-40}"
TOKEN_BUDGET="${TOKEN_BUDGET:-30000000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEQ_LEN="${SEQ_LEN:-2048}"
ARMS="${ARMS:-control delta vault phase}"
OUT_ROOT="${OUT_ROOT:-checkpoints_v11_recall_micro}"
LOG_ROOT="${LOG_ROOT:-logs/v11/recall_micro}"

arm_preset() {
  case "$1" in
    control) echo "v11_micro_10m" ;;
    delta)   echo "v11_micro_10m_delta" ;;
    vault)   echo "v11_micro_10m_vault" ;;
    phase)   echo "v11_micro_10m_phase" ;;
    *) echo "unknown arm: $1" >&2; exit 2 ;;
  esac
}

for arm in $ARMS; do
  preset="$(arm_preset "$arm")"
  ckpt_dir="$OUT_ROOT/$arm"
  log_dir="$LOG_ROOT/${arm}_$(date -u +%Y%m%d_%H%M%S)"
  mkdir -p "$ckpt_dir" "$log_dir"
  log="$log_dir/train.log"

  if [[ -f "$ckpt_dir/eval/verdict.json" ]]; then
    echo "[micro] skip $arm (verdict exists)" | tee -a "$LOG_ROOT/master.log"
    continue
  fi

  if [[ ! -f "$ckpt_dir/best_model.pt" && ! -f "$ckpt_dir/final_model.pt" && ! -f "$ckpt_dir/latest.pt" ]]; then
    echo "[micro] train $arm preset=$preset budget=$TOKEN_BUDGET" | tee -a "$LOG_ROOT/master.log"
    (
      set +e
      uv run python -m v11.train \
        --preset "$preset" --stage pretrain --dataset pretrain_mix --seq_len "$SEQ_LEN" \
        --batch_size "$BATCH_SIZE" --epochs 9999 --chunk_size 64 \
        --token_budget "$TOKEN_BUDGET" \
        --pretrain_sources recall --pretrain_weights 1 \
        --fineweb_name sample-10BT --blend_warmup_tokens 0 \
        --seed 42 --lr 3e-4 --warmup_steps 200 \
        --amp_dtype auto --num_workers 0 --gen_every 0 --save_every_steps 1000 \
        --no_grad_ckpt --compile --compile_mode default --fused_ce --fused_ce_chunk 4096 \
        --gate_surprisal_lambda 0.3 --gate_surprisal_tau 0.5 --gate_surprisal_sign 1.0 \
        --log_dir "$log_dir" --checkpoint_dir "$ckpt_dir"
      exit 0
    ) 2>&1 | tee -a "$log"
  else
    echo "[micro] train skip $arm (checkpoint exists)" | tee -a "$LOG_ROOT/master.log"
  fi

  ckpt=""
  for c in best_model.pt final_model.pt latest.pt; do
    [[ -f "$ckpt_dir/$c" ]] && ckpt="$ckpt_dir/$c" && break
  done
  if [[ -n "$ckpt" ]]; then
    echo "[micro] eval $arm" | tee -a "$LOG_ROOT/master.log"
    LABEL="$arm" PRESET="$preset" BEHAVIOR_TRIALS="$BEHAVIOR_TRIALS" \
      ./scripts/eval_recall_gate.sh "$ckpt" "$ckpt_dir/eval" "$preset" \
      2>&1 | tee -a "$log" || true
  else
    echo "[micro] FAIL $arm: no checkpoint" | tee -a "$LOG_ROOT/master.log"
  fi
done

# Tiny matched Mamba micro (same budget / recall-only data) if transformers available
mamba_dir="$OUT_ROOT/tiny_mamba"
if [[ ! -f "$mamba_dir/train_meta.json" ]]; then
  echo "[micro] train tiny_mamba" | tee -a "$LOG_ROOT/master.log"
  mkdir -p "$mamba_dir"
  (
    set +e
    # Reuse matched trainer with smaller budget; full mamba-130m is heavy for micro —
    # skip if MICRO_SKIP_MAMBA=1
    if [[ "${MICRO_SKIP_MAMBA:-0}" == "1" ]]; then
      echo "[micro] skip tiny_mamba (MICRO_SKIP_MAMBA=1)"
    else
      uv run python scripts/train_matched_baseline.py \
        --arch mamba --size tiny --token_budget "$TOKEN_BUDGET" --batch_size 16 --seq_len "$SEQ_LEN" \
        --pretrain_sources recall --pretrain_weights 1 \
        --checkpoint_dir "$mamba_dir" --lr 3e-4 --warmup_steps 200 \
        --save_every_steps 1000
    fi
    exit 0
  ) 2>&1 | tee -a "$LOG_ROOT/tiny_mamba.log"
fi

if [[ -f "$mamba_dir/best_hf/config.json" && ! -f "$mamba_dir/eval/mamba_behavior.json" ]]; then
  mkdir -p "$mamba_dir/eval"
  uv run python scripts/run_memory_behavioral.py \
    --model-type hf --model-id "$mamba_dir/best_hf" \
    --output "$mamba_dir/eval/mamba_behavior.json" \
    --context-lengths 128,512,1024,2048 --positions 0,0.5,1 \
    --association-counts 1,4,8 --trials "$BEHAVIOR_TRIALS" \
    2>&1 | tee -a "$LOG_ROOT/tiny_mamba.log" || true
fi

# Summarize
uv run python - <<'PY' | tee "$LOG_ROOT/summary.json"
import json
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
from behavioral_summary import behavioral_summary

root = Path('checkpoints_v11_recall_micro')
rows = []
for arm_dir in sorted(root.iterdir()):
    if not arm_dir.is_dir():
        continue
    v = arm_dir / 'eval' / 'verdict.json'
    bpath = arm_dir / 'eval' / 'v11_behavior.json'
    row = {'arm': arm_dir.name}
    if v.exists():
        data = json.loads(v.read_text())
        row.update({
            'recall_at_2048': data['behavioral'].get('single_assoc_at_max_context'),
            'multi8_at_min': data['behavioral'].get('multi8_at_min_context'),
            'gate': data['gate'].get('abs_content_minus_filler'),
            'ship': data.get('ship'),
        })
    elif bpath.exists():
        b = behavioral_summary(json.loads(bpath.read_text()))
        row.update({
            'recall_at_2048': b.get('single_assoc_at_max_context'),
            'multi8_at_min': b.get('multi8_at_min_context'),
        })
    rows.append(row)
print(json.dumps({'arms': rows}, indent=2))
PY

echo "[micro] done"
