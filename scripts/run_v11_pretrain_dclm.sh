#!/usr/bin/env bash
# Phase C pretrain: DCLM-Edu stream on top of v11_e3_k3 WikiText checkpoint.
#
# Usage:
#   ./scripts/run_v11_pretrain_dclm.sh [token_budget] [extra args]
#
# Examples:
#   ./scripts/run_v11_pretrain_dclm.sh                 # default 2B tokens
#   ./scripts/run_v11_pretrain_dclm.sh 500000000       # 500M smoke
#   tmux new-session -d -s v11_pretrain './scripts/run_v11_pretrain_dclm.sh'
#
#   # resume after crash (full state from latest.pt):
#   RESUME=checkpoints_v11_e3_k3_dclm/latest.pt \
#     tmux new-session -d -s v11_pretrain './scripts/run_v11_pretrain_dclm.sh'

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET="${PRESET:-v11_e3_k3}"
TOKEN_BUDGET="${1:-2000000000}"
shift || true

RESUME_FROM="${RESUME_FROM:-checkpoints_v11_e3_k3/best_model.pt}"
RESUME="${RESUME:-}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-5000}"
SEQ_LEN=2048
BATCH_SIZE=18
CHUNK_SIZE=256
EPOCHS=9999
EDU_SCORE_MIN=3
CKPT_DIR="${CKPT_DIR:-checkpoints_v11_e3_k3_dclm}"
EXTRA_ARGS="$*"

GEN_PROMPT="In 1923 , the University of"
LOG_DIR=$(make_log_dir "v11" "${PRESET}_pretrain_dclm_b${TOKEN_BUDGET}")
mkdir -p "$CKPT_DIR"

ARGS="--preset $PRESET --stage pretrain --dataset dclm_edu --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE --epochs $EPOCHS --chunk_size $CHUNK_SIZE \
  --token_budget $TOKEN_BUDGET --edu_score_min $EDU_SCORE_MIN \
  --resume_from $RESUME_FROM --amp_dtype auto --num_workers 0 \
  --gen_every 5000 --save_every_steps $SAVE_EVERY_STEPS --no_grad_ckpt \
  --compile --compile_mode default"

if [[ -n "$RESUME" ]]; then
  ARGS="$ARGS --resume $RESUME"
fi

RUN_DESC="V11 Phase C pretrain: $PRESET on DCLM-Edu (budget=${TOKEN_BUDGET} tok)"
write_run_info "$LOG_DIR" "$RUN_DESC" "$ARGS --gen_prompt '$GEN_PROMPT' $EXTRA_ARGS"

echo "============================================================"
echo "  V11 Phase C pretrain"
echo "  preset=$PRESET  budget=$TOKEN_BUDGET  resume=$RESUME_FROM"
echo "  Log: $LOG_DIR   Ckpt: $CKPT_DIR"
echo "============================================================"

eval "$PYTHON_BIN -m v11.train" \
  $ARGS \
  --gen_prompt "'$GEN_PROMPT'" \
  --log_dir "$LOG_DIR" \
  --checkpoint_dir "$CKPT_DIR" \
  $EXTRA_ARGS
