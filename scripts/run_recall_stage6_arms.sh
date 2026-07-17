#!/usr/bin/env bash
# Stage-6c: 300M-token architecture arms (control / delta / vault / phase / combo).
# Uses proven gate lever (λ=0.3, τ=0.5) + recall_w3 mix. Survivors of micro preferred.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_stage6 checkpoints_v11_recall_stage6

export FINEWEB_LOCAL_DIR="${FINEWEB_LOCAL_DIR:-data/fineweb-edu/sample-10BT}"
export HF_HUB_DISABLE_XET=1
export BEHAVIOR_TRIALS="${BEHAVIOR_TRIALS:-60}"

TOKEN_BUDGET="${TOKEN_BUDGET:-300000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GSL="${GSL:-0.3}"
GST="${GST:-0.5}"
GSSIGN="${GSSIGN:-1.0}"
RECALL_WEIGHT="${RECALL_WEIGHT:-3}"
WEB_WEIGHT=$((96 - RECALL_WEIGHT))
ARMS="${ARMS:-control delta vault phase}"
# Optional: ARMS="control delta vault phase combo" after pickers decide
OUT_ROOT="${OUT_ROOT:-checkpoints_v11_recall_stage6}"
LOG_ROOT="${LOG_ROOT:-logs/v11/recall_stage6}"
PRESET="${PRESET:-v11_e3_k3_chat}"

arm_extra() {
  case "$1" in
    control) echo "" ;;
    # K=1 delta: no E3 routing levers (they need n_states>1). Smaller delta_chunk for VRAM/stability.
    delta)   echo "--write_mode delta --n_states 1 --delta_chunk 32 --no_state_compete --no_routing_content_aware" ;;
    vault)   echo "--vault_state --vault_state_idx 0 --n_states 3" ;;
    phase)   echo "--write_phase_address --n_states 3" ;;
    combo)
      # Best combo of levers that beat control in micro / individual arms
      echo "--vault_state --vault_state_idx 0 --write_phase_address --n_states 3 ${COMBO_EXTRA:-}"
      ;;
    *) echo "unknown arm: $1" >&2; exit 2 ;;
  esac
}

for arm in $ARMS; do
  ckpt_dir="$OUT_ROOT/$arm"
  log_dir="$LOG_ROOT/${arm}_$(date -u +%Y%m%d_%H%M%S)"
  mkdir -p "$ckpt_dir" "$log_dir"
  log="$log_dir/train.log"
  extra="$(arm_extra "$arm")"

  if [[ -f "$ckpt_dir/eval/verdict.json" ]]; then
    echo "[stage6] skip $arm (verdict exists)" | tee -a "$LOG_ROOT/master.log"
    continue
  fi

  # Drop corrupt partial checkpoints from a prior disk-full / CUDA crash
  if [[ ! -f "$ckpt_dir/eval/verdict.json" ]]; then
    for c in best_model.pt final_model.pt latest.pt; do
      [[ -f "$ckpt_dir/$c" ]] || continue
      if ! uv run python -c "import torch; torch.load('$ckpt_dir/$c', map_location='cpu', weights_only=False)" >/dev/null 2>&1; then
        echo "[stage6] remove corrupt $ckpt_dir/$c" | tee -a "$LOG_ROOT/master.log"
        rm -f "$ckpt_dir/$c"
      fi
    done
  fi

  if [[ ! -f "$ckpt_dir/best_model.pt" && ! -f "$ckpt_dir/final_model.pt" && ! -f "$ckpt_dir/latest.pt" ]]; then
    echo "[stage6] train $arm extra=[$extra] budget=$TOKEN_BUDGET" | tee -a "$LOG_ROOT/master.log"
    # Delta: compile can still trip device asserts on some shapes — allow DELTA_NO_COMPILE=1
    compile_flags=(--compile --compile_mode default)
    if [[ "$arm" == "delta" && "${DELTA_NO_COMPILE:-0}" == "1" ]]; then
      compile_flags=()
      echo "[stage6] delta: training without torch.compile" | tee -a "$LOG_ROOT/master.log"
    fi
    # shellcheck disable=SC2086
    (
      set +e
      uv run python -m v11.train \
        --preset "$PRESET" --stage pretrain --dataset pretrain_mix --seq_len 2048 \
        --batch_size "$BATCH_SIZE" --epochs 9999 --chunk_size 256 \
        --token_budget "$TOKEN_BUDGET" --edu_score_min 3 \
        --pretrain_sources fineweb,recall --pretrain_weights "${WEB_WEIGHT},${RECALL_WEIGHT}" \
        --fineweb_name sample-10BT --blend_warmup_tokens 0 \
        --seed 42 --lr 1e-4 --warmup_steps 500 \
        --amp_dtype auto --num_workers 0 --gen_every 0 --save_every_steps 2000 \
        --no_grad_ckpt "${compile_flags[@]}" --fused_ce --fused_ce_chunk 4096 \
        --gate_surprisal_lambda "$GSL" --gate_surprisal_tau "$GST" --gate_surprisal_sign "$GSSIGN" \
        --state_compete --routing_content_aware --route_balance_lambda 0.01 \
        $extra \
        --log_dir "$log_dir" --checkpoint_dir "$ckpt_dir"
      exit 0
    ) 2>&1 | tee -a "$log"
  else
    echo "[stage6] train skip $arm (checkpoint exists)" | tee -a "$LOG_ROOT/master.log"
  fi

  ckpt=""
  for c in best_model.pt final_model.pt latest.pt; do
    [[ -f "$ckpt_dir/$c" ]] && ckpt="$ckpt_dir/$c" && break
  done
  if [[ -n "$ckpt" ]]; then
    # Delta arm uses n_states=1 — pass matching preset override via flags already in ckpt config
    LABEL="$arm" BEHAVIOR_TRIALS="$BEHAVIOR_TRIALS" \
      ./scripts/eval_recall_gate.sh "$ckpt" "$ckpt_dir/eval" "$PRESET" \
      2>&1 | tee -a "$log" || true
  else
    echo "[stage6] FAIL $arm: no checkpoint" | tee -a "$LOG_ROOT/master.log"
  fi
done

uv run python - <<'PY' | tee "$LOG_ROOT/summary.json"
import json
from pathlib import Path
root = Path('checkpoints_v11_recall_stage6')
rows = []
for d in sorted(root.iterdir()):
    v = d / 'eval' / 'verdict.json'
    if not v.exists():
        rows.append({'arm': d.name, 'status': 'pending'})
        continue
    data = json.loads(v.read_text())
    b, g = data['behavioral'], data['gate']
    rows.append({
        'arm': d.name,
        'recall_at_2048': b.get('single_assoc_at_max_context'),
        'multi8_at_min': b.get('multi8_at_min_context'),
        'gate': g.get('abs_content_minus_filler'),
        'ship': data.get('ship'),
    })
rows_sorted = sorted(
    [r for r in rows if r.get('recall_at_2048') is not None],
    key=lambda r: (-(r['recall_at_2048'] or 0), -(r.get('multi8_at_min') or 0)),
)
out = {'arms': rows, 'ranked': [r['arm'] for r in rows_sorted],
       'winner': rows_sorted[0]['arm'] if rows_sorted else None}
print(json.dumps(out, indent=2))
Path('logs/v11/recall_stage6/summary.json').write_text(json.dumps(out, indent=2)+'\n')
PY

echo "[stage6] done"
