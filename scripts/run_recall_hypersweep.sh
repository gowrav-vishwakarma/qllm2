#!/usr/bin/env bash
# Stage-3 hyper sweeps: warm-start retune to pick gate λ/τ, γ_floor knee, recall %.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_hypersweep checkpoints_v11_recall_hypersweep

LOG="logs/v11/recall_hypersweep/sweep.log"
: > "$LOG"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
PRESET="${PRESET:-v11_e3_k3_chat}"
TOKEN_BUDGET="${TOKEN_BUDGET:-150000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-2048}"
RESUME="${RESUME:-checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt}"
FINEWEB_LOCAL_DIR="${FINEWEB_LOCAL_DIR:-data/fineweb-edu/sample-10BT}"
OUT_ROOT="${OUT_ROOT:-checkpoints_v11_recall_hypersweep}"

export HF_HUB_DISABLE_XET=1
export FINEWEB_LOCAL_DIR

train_and_eval() {
  local arm="$1"
  shift
  local extra_flags=("$@")
  local recall_weight="${RECALL_WEIGHT:-0}"
  local sources="fineweb"
  local weights="96"
  if [[ "$recall_weight" -gt 0 ]]; then
    sources="fineweb,recall"
    weights="96,${recall_weight}"
  fi

  local ckpt_dir="$OUT_ROOT/$arm"
  local log_dir="logs/v11/recall_hypersweep/${arm}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$ckpt_dir"

  echo "================================================================" | tee -a "$LOG"
  echo "[hypersweep] arm=$arm sources=$sources weights=$weights flags=${extra_flags[*]}" | tee -a "$LOG"

  $PYTHON_BIN -m v11.train \
    --preset "$PRESET" --stage pretrain --dataset pretrain_mix --seq_len "$SEQ_LEN" \
    --batch_size "$BATCH_SIZE" --epochs 9999 --chunk_size 256 \
    --token_budget "$TOKEN_BUDGET" --edu_score_min 3 \
    --pretrain_sources "$sources" --pretrain_weights "$weights" \
    --fineweb_name sample-10BT --blend_warmup_tokens 0 \
    --seed 42 --lr 1e-4 --warmup_steps 500 \
    --amp_dtype auto --num_workers 0 --gen_every 0 --save_every_steps 2000 \
    --no_grad_ckpt --compile --compile_mode default --fused_ce --fused_ce_chunk 4096 \
    --resume_from "$RESUME" --no_resume_cursor \
    "${extra_flags[@]}" \
    --log_dir "$log_dir" --checkpoint_dir "$ckpt_dir" 2>&1 | tee -a "$LOG"

  local best="$ckpt_dir/best_model.pt"
  [[ -f "$best" ]] || best="$ckpt_dir/latest.pt"
  LABEL="$arm" ./scripts/eval_recall_gate.sh "$best" "$ckpt_dir/eval" "$PRESET" 2>&1 | tee -a "$LOG" || true
}

# Gate lambda / tau
for gsl in 0.1 0.3 0.5; do
  for gst in 0.5 1.0; do
    train_and_eval "gate_l${gsl}_t${gst}" \
      --gate_surprisal_lambda "$gsl" --gate_surprisal_tau "$gst" --gate_surprisal_sign 1.0
  done
done

# gamma_floor knee
for gf in 0.90 0.93 0.95 0.97; do
  train_and_eval "floor_g${gf}" --gamma_floor "$gf"
done

# Recall blend %
for rw in 3 10 20; do
  RECALL_WEIGHT="$rw" train_and_eval "recall_w${rw}"
done

echo "[hypersweep] all arms done" | tee -a "$LOG"
uv run python scripts/pick_recall_hypers.py \
  --root "$OUT_ROOT" --baseline-ppl 39.75 --ppl-tol 0.05 \
  --out logs/v11/recall_hypersweep/best_hypers.json | tee -a "$LOG"
