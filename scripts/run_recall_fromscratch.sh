#!/usr/bin/env bash
# Stage-3 from-scratch run: routing levers + best hypers from hypersweep.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_fromscratch checkpoints_v11_recall_fromscratch

HYPERS_JSON="${HYPERS_JSON:-logs/v11/recall_hypersweep/best_hypers.json}"
LOG="logs/v11/recall_fromscratch/train.log"
CKPT_DIR="checkpoints_v11_recall_fromscratch/best_run"

if [[ -f "$CKPT_DIR/eval/verdict.json" ]]; then
  echo "[fromscratch] SKIP (verdict exists)" | tee -a "$LOG"
  exit 0
fi

GSL="${GSL:-0.1}"
GST="${GST:-1.0}"
GSSIGN="${GSSIGN:-1.0}"
GFLOOR="${GFLOOR:-0.0}"
RECALL_WEIGHT="${RECALL_WEIGHT:-10}"
WEB_WEIGHT="${WEB_WEIGHT:-86}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1000000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FINEWEB_LOCAL_DIR="${FINEWEB_LOCAL_DIR:-data/fineweb-edu}"

if [[ -f "$HYPERS_JSON" ]]; then
  echo "[fromscratch] loading hypers from $HYPERS_JSON" | tee -a "$LOG"
  read -r GSL GST GFLOOR RECALL_WEIGHT <<<"$(uv run python - <<PY
import json
d=json.load(open('$HYPERS_JSON'))
r=d.get('recommended',{})
print(r.get('gate_surprisal_lambda',0.1), r.get('gate_surprisal_tau',1.0),
      r.get('gamma_floor',0.0), r.get('recall_weight',10))
PY
)"
  GSSIGN=1.0
  WEB_WEIGHT=$((96 - RECALL_WEIGHT))
  echo "[fromscratch] GSL=$GSL GST=$GST GFLOOR=$GFLOOR RECALL=$RECALL_WEIGHT WEB=$WEB_WEIGHT" | tee -a "$LOG"
else
  echo "[fromscratch] no $HYPERS_JSON; using Stage-2 defaults" | tee -a "$LOG"
fi

export HF_HUB_DISABLE_XET=1
export FINEWEB_LOCAL_DIR

EXTRA=(--state_compete --routing_content_aware --route_balance_lambda 0.01)
if uv run python -c "import sys; sys.exit(0 if float('${GSL}')>0 else 1)"; then
  EXTRA+=(--gate_surprisal_lambda "$GSL" --gate_surprisal_tau "$GST" --gate_surprisal_sign "$GSSIGN")
fi
if uv run python -c "import sys; sys.exit(0 if float('${GFLOOR}')>0 else 1)"; then
  EXTRA+=(--gamma_floor "$GFLOOR")
fi

BEST="$CKPT_DIR/best_model.pt"
[[ -f "$BEST" ]] || BEST="$CKPT_DIR/final_model.pt"
[[ -f "$BEST" ]] || BEST="$CKPT_DIR/latest.pt"

if [[ ! -f "$BEST" ]]; then
  LOG_DIR="logs/v11/recall_fromscratch/train_$(date +%Y%m%d_%H%M%S)"
  echo "[fromscratch] token_budget=$TOKEN_BUDGET routing=compete+content_aware" | tee -a "$LOG"
  echo "[fromscratch] flags: ${EXTRA[*]}" | tee -a "$LOG"
  (
    set +e
    uv run python -m v11.train \
      --preset v11_e3_k3_chat --stage pretrain --dataset pretrain_mix --seq_len 2048 \
      --batch_size "$BATCH_SIZE" --epochs 9999 --chunk_size 256 \
      --token_budget "$TOKEN_BUDGET" --edu_score_min 3 \
      --pretrain_sources fineweb,recall --pretrain_weights "${WEB_WEIGHT},${RECALL_WEIGHT}" \
      --fineweb_name sample-10BT --blend_warmup_tokens 0 \
      --seed 42 --lr 1e-4 --warmup_steps 500 \
      --amp_dtype auto --num_workers 0 --gen_every 0 --save_every_steps 2000 \
      --no_grad_ckpt --compile --compile_mode default --fused_ce --fused_ce_chunk 4096 \
      "${EXTRA[@]}" \
      --log_dir "$LOG_DIR" --checkpoint_dir "$CKPT_DIR"
    exit 0
  ) 2>&1 | tee -a "$LOG"
  BEST="$CKPT_DIR/best_model.pt"
  [[ -f "$BEST" ]] || BEST="$CKPT_DIR/final_model.pt"
  [[ -f "$BEST" ]] || BEST="$CKPT_DIR/latest.pt"
fi

if [[ -f "$BEST" ]]; then
  LABEL=fromscratch ./scripts/eval_recall_gate.sh "$BEST" "$CKPT_DIR/eval" v11_e3_k3_chat 2>&1 | tee -a "$LOG" || true
  echo "[fromscratch] done -> $CKPT_DIR/eval/verdict.json" | tee -a "$LOG"
else
  echo "[fromscratch] FAIL — no checkpoint" | tee -a "$LOG"
  exit 1
fi
