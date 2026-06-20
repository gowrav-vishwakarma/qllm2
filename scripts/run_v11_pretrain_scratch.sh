#!/usr/bin/env bash
# Phase 2 knowledge pretrain: FROM SCRATCH, chat vocab (50259) baked in,
# mixed web-edu corpus (DCLM-Edu + FineWeb-Edu), large token budget, cosine LR.
#
# This is the production "better base" run. It is resumable: re-run the SAME
# command with RESUME=<ckpt> to continue (restores optimizer + scheduler + step
# + token count). All hyperparameters live here so a restart never drifts.
#
# Usage (persistent, survives disconnect):
#   tmux new-session -d -s v11_pretrain './scripts/run_v11_pretrain_scratch.sh'
#   tmux attach -t v11_pretrain
#
#   # custom budget (e.g. 20B tokens):
#   tmux new-session -d -s v11_pretrain './scripts/run_v11_pretrain_scratch.sh 20000000000'
#
#   # resume after a crash / manual stop:
#   RESUME=checkpoints_v11_e3_k3_chat_pretrain/best_model.pt \
#     tmux new-session -d -s v11_pretrain './scripts/run_v11_pretrain_scratch.sh'

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Hyperparameters (single source of truth; restart-safe) ───────────────────
PRESET="${PRESET:-v11_e3_k3_chat}"          # vocab 50259 baked in at init
TOKEN_BUDGET="${1:-10000000000}"            # ~10B tokens to start (extensible)
shift || true

DATASET="${DATASET:-pretrain_mix}"
PRETRAIN_SOURCES="${PRETRAIN_SOURCES:-dclm,fineweb}"
PRETRAIN_WEIGHTS="${PRETRAIN_WEIGHTS:-1,1}"
FINEWEB_NAME="${FINEWEB_NAME:-sample-10BT}"
EDU_SCORE_MIN="${EDU_SCORE_MIN:-3}"

SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-18}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
EPOCHS=9999                                  # budget-bound, not epoch-bound
LR="${LR:-3e-4}"                             # from-scratch peak; cosine to budget
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

CKPT_DIR="${CKPT_DIR:-checkpoints_v11_e3_k3_chat_pretrain}"
RESUME="${RESUME:-}"                         # set to a ckpt path to continue
EXTRA_ARGS="$*"

GEN_PROMPT="In 1923 , the University of"
LOG_DIR=$(make_log_dir "v11" "${PRESET}_pretrain_scratch_b${TOKEN_BUDGET}")
mkdir -p "$CKPT_DIR"

ARGS="--preset $PRESET --stage pretrain --dataset $DATASET --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE --epochs $EPOCHS --chunk_size $CHUNK_SIZE \
  --token_budget $TOKEN_BUDGET --edu_score_min $EDU_SCORE_MIN \
  --pretrain_sources $PRETRAIN_SOURCES --pretrain_weights $PRETRAIN_WEIGHTS \
  --fineweb_name $FINEWEB_NAME \
  --lr $LR --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
  --amp_dtype auto --num_workers 0 --gen_every 5000 --no_grad_ckpt \
  --compile --compile_mode default"

if [[ -n "$RESUME" ]]; then
  ARGS="$ARGS --resume $RESUME"
fi

RUN_DESC="V11 Phase 2 from-scratch pretrain: $PRESET on $DATASET ($PRETRAIN_SOURCES), budget=${TOKEN_BUDGET} tok"
write_run_info "$LOG_DIR" "$RUN_DESC" "$ARGS --gen_prompt '$GEN_PROMPT' $EXTRA_ARGS"

echo "============================================================"
echo "  V11 Phase 2 from-scratch knowledge pretrain"
echo "  preset=$PRESET  budget=$TOKEN_BUDGET  sources=$PRETRAIN_SOURCES"
echo "  lr=$LR warmup=$WARMUP_STEPS  resume=${RESUME:-<scratch>}"
echo "  Log: $LOG_DIR   Ckpt: $CKPT_DIR"
echo "============================================================"

eval "$PYTHON_BIN -m v11.train" \
  $ARGS \
  --gen_prompt "'$GEN_PROMPT'" \
  --log_dir "$LOG_DIR" \
  --checkpoint_dir "$CKPT_DIR" \
  $EXTRA_ARGS
