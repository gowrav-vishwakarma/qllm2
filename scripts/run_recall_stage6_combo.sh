#!/usr/bin/env bash
# Stage-6d: 1B-token from-scratch confirmation of winning Stage-6 levers.
# Reads logs/v11/recall_stage6/summary.json (or env COMBO_EXTRA) for which levers to combine.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_stage6_combo checkpoints_v11_recall_stage6_combo

export FINEWEB_LOCAL_DIR="${FINEWEB_LOCAL_DIR:-data/fineweb-edu/sample-10BT}"
export HF_HUB_DISABLE_XET=1
export BEHAVIOR_TRIALS="${BEHAVIOR_TRIALS:-60}"

TOKEN_BUDGET="${TOKEN_BUDGET:-1000000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GSL="${GSL:-0.3}"
GST="${GST:-0.5}"
GSSIGN="${GSSIGN:-1.0}"
RECALL_WEIGHT="${RECALL_WEIGHT:-3}"
WEB_WEIGHT=$((96 - RECALL_WEIGHT))
CKPT_DIR="${CKPT_DIR:-checkpoints_v11_recall_stage6_combo/best_run}"
LOG_DIR="${LOG_DIR:-logs/v11/recall_stage6_combo/train_$(date -u +%Y%m%d_%H%M%S)}"
PRESET="${PRESET:-v11_e3_k3_chat}"

# Pick combo extras from stage6 summary if present
if [[ -z "${COMBO_EXTRA:-}" && -f logs/v11/recall_stage6/summary.json ]]; then
  read -r COMBO_EXTRA <<<"$(uv run python - <<'PY'
import json
from pathlib import Path
s = json.loads(Path('logs/v11/recall_stage6/summary.json').read_text())
arms = {r['arm']: r for r in s.get('arms', []) if r.get('recall_at_2048') is not None}
ctrl = arms.get('control', {}).get('recall_at_2048', 0.0) or 0.0
extras = []
# Include lever if it beats control by >0.05 (beyond ~noise with n=60)
for name, flag in [('vault', '--vault_state --vault_state_idx 0'),
                   ('phase', '--write_phase_address'),
                   ('delta', '--write_mode delta --n_states 1 --delta_chunk 64')]:
    r = arms.get(name, {}).get('recall_at_2048')
    m8 = arms.get(name, {}).get('multi8_at_min') or 0
    m8c = arms.get('control', {}).get('multi8_at_min') or 0
    if r is not None and (r >= ctrl + 0.05 or m8 >= m8c + 0.05):
        extras.append(flag)
# Prefer vault+phase over delta if both win (delta is K=1, incompatible with vault)
if any('vault' in e or 'phase' in e for e in extras):
    extras = [e for e in extras if 'write_mode delta' not in e]
if not extras:
    # Fall back to vault+phase (architecture thesis) if nothing clearly wins
    extras = ['--vault_state --vault_state_idx 0 --write_phase_address']
print(' '.join(extras))
PY
)"
fi
COMBO_EXTRA="${COMBO_EXTRA:---vault_state --vault_state_idx 0 --write_phase_address}"

if [[ -f "$CKPT_DIR/eval/verdict.json" ]]; then
  echo "[combo] skip (verdict exists)"
  exit 0
fi

mkdir -p "$CKPT_DIR" "$LOG_DIR"
echo "[combo] extras=[$COMBO_EXTRA] budget=$TOKEN_BUDGET" | tee -a logs/v11/recall_stage6_combo/master.log

if [[ ! -f "$CKPT_DIR/best_model.pt" && ! -f "$CKPT_DIR/final_model.pt" ]]; then
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
      --no_grad_ckpt --compile --compile_mode default --fused_ce --fused_ce_chunk 4096 \
      --gate_surprisal_lambda "$GSL" --gate_surprisal_tau "$GST" --gate_surprisal_sign "$GSSIGN" \
      --state_compete --routing_content_aware --route_balance_lambda 0.01 \
      $COMBO_EXTRA \
      --log_dir "$LOG_DIR" --checkpoint_dir "$CKPT_DIR"
    exit 0
  ) 2>&1 | tee -a "$LOG_DIR/train.log"
fi

ckpt=""
for c in best_model.pt final_model.pt latest.pt; do
  [[ -f "$CKPT_DIR/$c" ]] && ckpt="$CKPT_DIR/$c" && break
done
if [[ -n "$ckpt" ]]; then
  LABEL=stage6_combo BEHAVIOR_TRIALS="$BEHAVIOR_TRIALS" \
    ./scripts/eval_recall_gate.sh "$ckpt" "$CKPT_DIR/eval" "$PRESET"
fi
echo "[combo] done"
