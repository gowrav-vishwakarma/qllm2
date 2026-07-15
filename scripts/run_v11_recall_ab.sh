#!/usr/bin/env bash
# Recall-program A/B: train short matched V11 arms that isolate each recall lever,
# then run the eval gate (gate selectivity + behavioral recall) on each.
#
# All arms share preset v11_e3_k3_chat (same params/vocab), seed, and token budget;
# only the recall levers differ so differences are attributable. GPU required; run
# OUTSIDE the sandbox.
#
# Arms (space-separated env ARMS to override):
#   control  : stock architecture, web-only blend
#   gate     : + gate-surprisal aux loss
#   floor    : + gamma_floor (longer memory horizon)
#   recall   : + synthetic recall curriculum in the data blend
#   combo    : gate + floor + recall together
#
# Usage:
#   ./scripts/run_v11_recall_ab.sh                 # full default sweep
#   ARMS="control combo" TOKEN_BUDGET=60000000 ./scripts/run_v11_recall_ab.sh
#   DRY=1 ./scripts/run_v11_recall_ab.sh           # print commands only
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
PRESET="${PRESET:-v11_e3_k3_chat}"
SEED="${SEED:-42}"
TOKEN_BUDGET="${TOKEN_BUDGET:-300000000}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
EDU_MIN="${EDU_MIN:-3}"
FINEWEB_NAME="${FINEWEB_NAME:-sample-10BT}"
# gate-surprisal + gamma_floor hyperparameters (shared across arms that use them)
GSL="${GSL:-0.1}"
GST="${GST:-1.0}"
GSSIGN="${GSSIGN:-1.0}"
GFLOOR="${GFLOOR:-0.98}"
# recall data weight (% of blend) — web weights split the remainder evenly
RECALL_WEIGHT="${RECALL_WEIGHT:-3}"
WEB_WEIGHT="${WEB_WEIGHT:-48}"          # per web source (dclm, fineweb)
# Comma-separated web sources. Override to "fineweb" when DCLM is unavailable
# (e.g. this box has no HF auth and only local FineWeb shards).
WEB_SOURCES="${WEB_SOURCES:-dclm,fineweb}"
OUT_ROOT="${OUT_ROOT:-checkpoints_v11_recall_ab}"
DRY="${DRY:-0}"
# Warm-start from an existing checkpoint (fast Stage-2 screen: measures the recall
# levers as a continued-pretraining delta). Empty => train from scratch.
RESUME="${RESUME:-}"
# NO_CURSOR=1: warm-start weights WITHOUT seeding the multi-million-doc skip cursor
# (fast startup; accepts some data overlap — fine for a relative A/B). Recommended
# for the warm_fast screen. Ignored when RESUME is empty.
NO_CURSOR="${NO_CURSOR:-1}"

ARMS="${ARMS:-control gate floor recall combo}"

arm_flags() {  # -> extra train flags for the arm
  case "$1" in
    control) echo "" ;;
    gate)    echo "--gate_surprisal_lambda $GSL --gate_surprisal_tau $GST --gate_surprisal_sign $GSSIGN" ;;
    floor)   echo "--gamma_floor $GFLOOR" ;;
    recall)  echo "" ;;  # data-only arm; sources set separately
    combo)   echo "--gate_surprisal_lambda $GSL --gate_surprisal_tau $GST --gate_surprisal_sign $GSSIGN --gamma_floor $GFLOOR" ;;
    *) echo "unknown arm: $1" >&2; exit 2 ;;
  esac
}

arm_sources() {  # -> (sources, weights) for the arm
  # Expand WEB_SOURCES into equal WEB_WEIGHT entries.
  IFS=',' read -r -a _webs <<< "$WEB_SOURCES"
  _web_srcs="$WEB_SOURCES"
  _web_wts=""
  for _w in "${_webs[@]}"; do
    [[ -n "$_web_wts" ]] && _web_wts="${_web_wts},"
    _web_wts="${_web_wts}${WEB_WEIGHT}"
  done
  case "$1" in
    recall|combo) echo "${_web_srcs},recall ${_web_wts},${RECALL_WEIGHT}" ;;
    *)            echo "${_web_srcs} ${_web_wts}" ;;
  esac
}

mkdir -p "$OUT_ROOT"
echo "[recall-ab] arms='$ARMS' budget=$TOKEN_BUDGET preset=$PRESET seed=$SEED out=$OUT_ROOT"

for arm in $ARMS; do
  CKPT_DIR="$OUT_ROOT/$arm"
  LOG_DIR="logs/v11/recall_ab_${arm}_$(date +%Y%m%d_%H%M%S)"
  read -r SRCS WTS <<<"$(arm_sources "$arm")"
  EXTRA="$(arm_flags "$arm")"
  if [[ -n "$RESUME" ]]; then
    EXTRA="$EXTRA --resume_from $RESUME"
    [[ "$NO_CURSOR" == "1" ]] && EXTRA="$EXTRA --no_resume_cursor"
  fi
  TRAIN_CMD="$PYTHON_BIN -m v11.train \
    --preset $PRESET --stage pretrain --dataset pretrain_mix --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE --epochs 9999 --chunk_size $CHUNK_SIZE \
    --token_budget $TOKEN_BUDGET --edu_score_min $EDU_MIN \
    --pretrain_sources $SRCS --pretrain_weights $WTS \
    --fineweb_name $FINEWEB_NAME --blend_warmup_tokens 0 \
    --seed $SEED --lr $LR --warmup_steps $WARMUP_STEPS \
    --amp_dtype auto --num_workers 0 --gen_every 0 --save_every_steps 2000 \
    --no_grad_ckpt --compile --compile_mode default --fused_ce --fused_ce_chunk 4096 \
    $EXTRA --log_dir $LOG_DIR --checkpoint_dir $CKPT_DIR"

  echo "=================================================================="
  echo "[arm:$arm] sources=$SRCS weights=$WTS flags='${EXTRA:-<none>}'"
  echo "[arm:$arm] ckpt=$CKPT_DIR log=$LOG_DIR"
  if [[ "$DRY" == "1" ]]; then
    echo "[DRY] $TRAIN_CMD"
    continue
  fi
  mkdir -p "$CKPT_DIR"
  eval "$TRAIN_CMD"

  BEST="$CKPT_DIR/best_model.pt"
  [[ -f "$BEST" ]] || BEST="$CKPT_DIR/latest.pt"
  echo "[arm:$arm] eval gate on $BEST"
  LABEL="$arm" ./scripts/eval_recall_gate.sh "$BEST" "$OUT_ROOT/$arm/eval" "$PRESET" || true
done

echo "[recall-ab] all arms done. Verdicts:"
for arm in $ARMS; do
  V="$OUT_ROOT/$arm/eval/verdict.json"
  [[ -f "$V" ]] && echo "  $arm -> $V"
done
