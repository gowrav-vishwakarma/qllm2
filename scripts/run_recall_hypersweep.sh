#!/usr/bin/env bash
# Stage-3 hyper sweeps: warm-start retune to pick gate λ/τ, γ_floor knee, recall %.
# Resumable: skips arms that already have eval/verdict.json.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_hypersweep checkpoints_v11_recall_hypersweep

LOG="logs/v11/recall_hypersweep/sweep.log"
PRESET="${PRESET:-v11_e3_k3_chat}"
TOKEN_BUDGET="${TOKEN_BUDGET:-150000000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-2048}"
RESUME="${RESUME:-checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt}"
FINEWEB_LOCAL_DIR="${FINEWEB_LOCAL_DIR:-data/fineweb-edu}"
OUT_ROOT="${OUT_ROOT:-checkpoints_v11_recall_hypersweep}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"

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
  mkdir -p "$ckpt_dir"

  if [[ -f "$ckpt_dir/eval/verdict.json" ]]; then
    echo "[hypersweep] SKIP $arm (verdict exists)" | tee -a "$LOG"
    return 0
  fi

  local best="$ckpt_dir/best_model.pt"
  [[ -f "$best" ]] || best="$ckpt_dir/final_model.pt"
  [[ -f "$best" ]] || best="$ckpt_dir/latest.pt"

  if [[ ! -f "$best" ]]; then
    local log_dir="logs/v11/recall_hypersweep/${arm}_$(date +%Y%m%d_%H%M%S)"
    echo "================================================================" | tee -a "$LOG"
    echo "[hypersweep] TRAIN arm=$arm sources=$sources weights=$weights flags=${extra_flags[*]}" | tee -a "$LOG"
    # Run in subshell; torch compile teardown can abort — don't kill the sweep.
    (
      set +e
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
        --log_dir "$log_dir" --checkpoint_dir "$ckpt_dir"
      exit 0
    ) 2>&1 | tee -a "$LOG"
    best="$ckpt_dir/best_model.pt"
    [[ -f "$best" ]] || best="$ckpt_dir/final_model.pt"
    [[ -f "$best" ]] || best="$ckpt_dir/latest.pt"
  else
    echo "[hypersweep] SKIP train $arm (checkpoint exists: $best)" | tee -a "$LOG"
  fi

  if [[ -f "$best" ]]; then
    echo "[hypersweep] EVAL arm=$arm on $best" | tee -a "$LOG"
    LABEL="$arm" ./scripts/eval_recall_gate.sh "$best" "$ckpt_dir/eval" "$PRESET" 2>&1 | tee -a "$LOG" || true
  else
    echo "[hypersweep] FAIL $arm — no checkpoint" | tee -a "$LOG"
  fi
}

echo "[hypersweep] $(date -u +%FT%TZ) start/resume" | tee -a "$LOG"

for gsl in 0.1 0.3 0.5; do
  for gst in 0.5 1.0; do
    train_and_eval "gate_l${gsl}_t${gst}" \
      --gate_surprisal_lambda "$gsl" --gate_surprisal_tau "$gst" --gate_surprisal_sign 1.0
  done
done

for gf in 0.90 0.93 0.95 0.97; do
  train_and_eval "floor_g${gf}" --gamma_floor "$gf"
done

for rw in 3 10 20; do
  RECALL_WEIGHT="$rw" train_and_eval "recall_w${rw}"
done

echo "[hypersweep] all arms done $(date -u +%FT%TZ)" | tee -a "$LOG"
$PYTHON_BIN scripts/pick_recall_hypers.py \
  --root "$OUT_ROOT" --baseline-ppl 39.75 --ppl-tol 0.05 \
  --out logs/v11/recall_hypersweep/best_hypers.json | tee -a "$LOG"
