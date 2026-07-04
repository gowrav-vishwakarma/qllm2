#!/usr/bin/env bash
# One shipping round for the v2 gate line (phase-aware gate, vocab 50261, blended
# pretrain). Single entrypoint for: pretrain -> probes -> sft -> smoke -> export
# on GCP, and ship (pull+verify+push) on the RTX4090.
#
# Freshness: cursors are auto-derived from the previous round's checkpoint
# (per_source_docs) via scripts/v11_data_cursor.py, so no web/reasoning doc is
# reused. The trainer also auto-seeds skips from --resume checkpoints.
#
# Usage (GCP, tmux):
#   # Round 1 from scratch (2B blended pretrain):
#   ROUND=1 ROUND_TAG=round-2b-gate SCRATCH=1 TOKEN_BUDGET=2000000000 \
#     tmux new-session -d -s v11_round './scripts/run_v11_round.sh pretrain'
#   ./scripts/run_v11_round.sh probes
#   ./scripts/run_v11_round.sh sft
#   ./scripts/run_v11_round.sh smoke
#   ROUND=1 ROUND_TAG=round-2b-gate TOKEN_BUDGET=2000000000 ./scripts/run_v11_round.sh export
#
#   # Round 2+ (resume previous pretrain best, fresh tokens):
#   ROUND=2 ROUND_TAG=round-4b-gate TOKEN_BUDGET=2000000000 ./scripts/run_v11_round.sh pretrain
#   ...
#
# Ship (RTX4090):
#   ROUND_TAG=round-2b-gate ./scripts/run_v11_round.sh ship

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

STEP="${1:-help}"; shift || true

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET="${PRESET:-v11_e3_k3_chat}"
ROUND="${ROUND:-1}"
ROUND_TAG="${ROUND_TAG:-round-${ROUND}}"
TOKEN_BUDGET="${TOKEN_BUDGET:-2000000000}"
SCRATCH="${SCRATCH:-0}"

PT_CKPT_DIR="${PT_CKPT_DIR:-checkpoints_v11_e3_k3_chat_pretrain_v2}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-checkpoints_v11_sft_chat_smoltalk_v2}"

# Blend: web-dominant + small reasoning/chat (smoltalk2 Mid). Weights are
# PER-DOCUMENT; Mid docs are ~10x larger, so realized reasoning token share is
# higher than the raw weight. Check per_source_tokens after round 1 and tune.
PRETRAIN_SOURCES="${PRETRAIN_SOURCES:-dclm,fineweb,smoltalk2_mid}"
PRETRAIN_WEIGHTS="${PRETRAIN_WEIGHTS:-48,48,4}"
FINEWEB_NAME="${FINEWEB_NAME:-sample-10BT}"
EDU_SCORE_MIN="${EDU_SCORE_MIN:-3}"
THINK_FRACTION="${THINK_FRACTION:-0.15}"

# Grammar warmup only matters from scratch (base has no grammar yet).
if [[ "$SCRATCH" == "1" ]]; then
  BLEND_WARMUP_TOKENS="${BLEND_WARMUP_TOKENS:-1000000000}"
else
  BLEND_WARMUP_TOKENS="${BLEND_WARMUP_TOKENS:-0}"
fi

SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-18}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
LR="${LR:-3e-4}"                 # from-scratch peak; lower (1e-4) for resume rounds
[[ "$SCRATCH" != "1" ]] && LR="${LR_RESUME:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
SEED="${SEED:-42}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-5000}"
COMPILE="${COMPILE:-1}"
MIN_DISK_GB="${MIN_DISK_GB:-15}"

COMPILE_ARGS="--compile --compile_mode default"
[[ "$COMPILE" == "0" ]] && COMPILE_ARGS=""

disk_check() {
  local free_gb
  free_gb="$(df -BG / | awk 'NR==2{gsub("G","",$4); print $4}')"
  echo "[disk] / free: ${free_gb}GB (require >= ${MIN_DISK_GB}GB)"
  if (( free_gb < MIN_DISK_GB )); then
    echo "[disk] ABORT: not enough free disk. Prune old checkpoints/logs." >&2
    exit 1
  fi
}

pt_best() { echo "${PT_CKPT_DIR}/best_model.pt"; }
sft_best() { echo "${SFT_CKPT_DIR}/best_model.pt"; }

case "$STEP" in
pretrain)
  disk_check
  mkdir -p "$PT_CKPT_DIR"
  RESUME_ARGS=""
  CURSOR_ARGS=""
  if [[ "$SCRATCH" == "1" ]]; then
    echo "[pretrain] from scratch (round $ROUND) -> $PT_CKPT_DIR"
  else
    RESUME_ARGS="--resume $(pt_best)"
    # Auto cursors from the previous round's checkpoint (fresh tokens).
    eval "$(eval "$PYTHON_BIN scripts/v11_data_cursor.py --checkpoint $(pt_best) --fineweb-name $FINEWEB_NAME --emit env")"
    CURSOR_ARGS="--dclm_skip_docs ${DCLM_SKIP_DOCS:-0} --fineweb_skip_docs ${FINEWEB_SKIP_DOCS:-0} --smoltalk2_mid_skip_rows ${SMOLTALK2_MID_SKIP_ROWS:-0}"
    FINEWEB_NAME="${FINEWEB_NAME:-sample-10BT}"
    echo "[pretrain] resume $(pt_best) cursors: $CURSOR_ARGS fineweb=$FINEWEB_NAME"
  fi
  LOG_DIR=$(make_log_dir "v11" "round${ROUND}_pretrain")
  GEN_PROMPT="In 1923 , the University of"
  ARGS="--preset $PRESET --stage pretrain --dataset pretrain_mix --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE --epochs 9999 --chunk_size $CHUNK_SIZE \
    --token_budget $TOKEN_BUDGET --edu_score_min $EDU_SCORE_MIN \
    --pretrain_sources $PRETRAIN_SOURCES --pretrain_weights $PRETRAIN_WEIGHTS \
    --fineweb_name $FINEWEB_NAME --blend_warmup_tokens $BLEND_WARMUP_TOKENS \
    --seed $SEED --lr $LR --warmup_steps $WARMUP_STEPS \
    --amp_dtype auto --num_workers 0 --gen_every 5000 --save_every_steps $SAVE_EVERY_STEPS \
    --no_grad_ckpt $COMPILE_ARGS $RESUME_ARGS $CURSOR_ARGS"
  write_run_info "$LOG_DIR" "V11 round $ROUND pretrain ($ROUND_TAG)" "$ARGS"
  eval "$PYTHON_BIN -m v11.train" $ARGS \
    --gen_prompt "'$GEN_PROMPT'" --log_dir "$LOG_DIR" --checkpoint_dir "$PT_CKPT_DIR"
  ;;

probes)
  echo "[probes] gate + rank on $(pt_best)"
  eval "$PYTHON_BIN scripts/v11_probe_gates.py --checkpoint $(pt_best) --preset $PRESET" || true
  eval "$PYTHON_BIN -m memory_probes --test rank-text --checkpoint $(pt_best) --preset $PRESET --layer 4 --text-tokens 5000 --sample-every 100" || true
  ;;

sft)
  disk_check
  mkdir -p "$SFT_CKPT_DIR"
  LOG_DIR=$(make_log_dir "v11" "round${ROUND}_sft")
  GEN_PROMPT="<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
  ARGS="--preset $PRESET --stage sft --dataset smoltalk2 --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE --epochs ${SFT_EPOCHS:-1} --chunk_size $CHUNK_SIZE \
    --lr ${SFT_LR:-5e-5} --warmup_steps 200 --sft_filter hard \
    --think_fraction $THINK_FRACTION --smoltalk2_skip_rows ${SMOLTALK2_SKIP_ROWS:-0} \
    ${SFT_MAX_SAMPLES:+--max_samples $SFT_MAX_SAMPLES} \
    --amp_dtype auto --num_workers 4 --gen_every 0 --no_grad_ckpt $COMPILE_ARGS \
    --resume_from $(pt_best)"
  write_run_info "$LOG_DIR" "V11 round $ROUND smoltalk2 SFT ($ROUND_TAG)" "$ARGS"
  eval "$PYTHON_BIN -m v11.train" $ARGS \
    --gen_prompt "'$GEN_PROMPT'" --log_dir "$LOG_DIR" --checkpoint_dir "$SFT_CKPT_DIR"
  ;;

smoke)
  echo "[smoke] $(sft_best)"
  eval "$PYTHON_BIN scripts/smoke_chat_v11.py --checkpoint $(sft_best) --preset $PRESET --label $ROUND_TAG"
  ;;

export)
  disk_check
  echo "[export] $(sft_best) -> hf_release + server_manifest ($ROUND_TAG)"
  eval "$PYTHON_BIN scripts/export_hf_release.py \
    --src $(sft_best) --round $ROUND_TAG --tag $ROUND_TAG \
    --pretrain_tokens_total ${PRETRAIN_TOKENS_TOTAL:-$TOKEN_BUDGET} \
    --round_tokens $TOKEN_BUDGET --record-manifest"
  # Keep a copy of the bundle under releases/<tag>/ so the pull script path matches.
  mkdir -p "releases/$ROUND_TAG"
  cp -f hf_release/qllm_v11_e3k3_chat.pt "releases/$ROUND_TAG/qllm_v11_e3k3_chat.pt"
  # Point the manifest bundle path at the per-round copy for pull.
  eval "$PYTHON_BIN - <<PY
import json
p='releases/server_manifest.json'; m=json.load(open(p))
m['rounds']['$ROUND_TAG']['hf_export_bundle']='releases/$ROUND_TAG/qllm_v11_e3k3_chat.pt'
import hashlib
m['rounds']['$ROUND_TAG']['sha256']=hashlib.sha256(open('releases/$ROUND_TAG/qllm_v11_e3k3_chat.pt','rb').read()).hexdigest()
json.dump(m, open(p,'w'), indent=2)
print('manifest bundle path + sha updated for $ROUND_TAG')
PY"
  echo "[export] done. Now ship from the RTX4090:  ROUND_TAG=$ROUND_TAG ./scripts/run_v11_round.sh ship"
  ;;

eval)
  echo "[eval] batch chat samples for $ROUND_TAG"
  tag="${ROUND_TAG:-round-2b-gate}"
  ( cd hf_release && eval "$PYTHON_BIN eval_chat.py" \
    --checkpoint qllm_v11_e3k3_chat.pt \
    --prompts eval_prompts_round1.yaml \
    --out-md "SAMPLES_${tag}.md" \
    --out-json "../logs/v11/${tag}_chat_eval.json" )
  ;;

ship)
  echo "[ship] pull -> verify -> push ($ROUND_TAG) [run on RTX4090 with hf auth]"
  ROUND="$ROUND_TAG" ./scripts/pull_v11_release.sh --round "$ROUND_TAG"
  cp -f "releases/$ROUND_TAG/qllm_v11_e3k3_chat.pt" hf_release/qllm_v11_e3k3_chat.pt
  ( cd hf_release && bash verify.sh && bash verify_legacy.sh )
  eval "$PYTHON_BIN scripts/push_qllm_hf.py --revision $ROUND_TAG"
  echo "[ship] published revision $ROUND_TAG"
  ;;

help|*)
  cat <<EOF
run_v11_round.sh <step>
  steps (GCP):  pretrain | probes | sft | smoke | export | eval
  step (4090):  ship
  key env:      ROUND ROUND_TAG SCRATCH TOKEN_BUDGET PRETRAIN_WEIGHTS
                BLEND_WARMUP_TOKENS THINK_FRACTION FINEWEB_NAME
EOF
  ;;
esac
