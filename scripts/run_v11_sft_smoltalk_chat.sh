#!/usr/bin/env bash
# Re-SFT: SmolTalk2 on the 10B from-scratch chat base (ChatML vocab 50259).
# Fixes the Tulu-3 regression by using clean chat data + warm-started ChatML rows.
#
# Usage (RTX box, tmux):
#   tmux new-session -d -s v11_sft_smoltalk \
#     './scripts/run_v11_sft_smoltalk_chat.sh'
#   tmux attach -t v11_sft_smoltalk
#
# Resume a crashed run:
#   RESUME=checkpoints_v11_sft_chat_smoltalk/best_model.pt \
#     tmux new-session -d -s v11_sft_smoltalk './scripts/run_v11_sft_smoltalk_chat.sh'

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET="${PRESET:-v11_e3_k3_chat}"
RESUME_FROM="${1:-checkpoints_v11_e3_k3_chat_pretrain/best_model.pt}"
shift || true

SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-18}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-5}"
SFT_FILTER="${SFT_FILTER:-hard}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
CKPT_DIR="${CKPT_DIR:-checkpoints_v11_sft_chat_smoltalk}"
RESUME="${RESUME:-}"
EXTRA_ARGS="$*"

GEN_PROMPT="<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
LOG_DIR=$(make_log_dir "v11" "${PRESET}_sft_smoltalk2_chat")
mkdir -p "$CKPT_DIR"

ARGS="--preset $PRESET --stage sft --dataset smoltalk2 --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE --epochs $EPOCHS --chunk_size $CHUNK_SIZE \
  --lr $LR --warmup_steps $WARMUP_STEPS --sft_filter $SFT_FILTER \
  --amp_dtype auto --num_workers 4 --gen_every 0 --no_grad_ckpt \
  --compile --compile_mode default --warmstart_chatml"

if [[ -n "$RESUME" ]]; then
  ARGS="$ARGS --resume $RESUME"
else
  ARGS="$ARGS --resume_from $RESUME_FROM"
fi

RUN_DESC="V11 SmolTalk2 re-SFT: $PRESET on chat base (warmstart ChatML)"
write_run_info "$LOG_DIR" "$RUN_DESC" "$ARGS $EXTRA_ARGS"

echo "============================================================"
echo "  V11 SmolTalk2 re-SFT (ChatML + warmstart)"
echo "  preset=$PRESET  base=$RESUME_FROM  epochs=$EPOCHS"
echo "  resume=${RESUME:-<none>}  ckpt=$CKPT_DIR"
echo "  Log: $LOG_DIR"
echo "============================================================"

eval "$PYTHON_BIN -m v11.train" \
  $ARGS \
  --gen_prompt "'$GEN_PROMPT'" \
  --log_dir "$LOG_DIR" \
  --checkpoint_dir "$CKPT_DIR" \
  $EXTRA_ARGS
