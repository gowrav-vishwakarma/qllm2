#!/usr/bin/env bash
# One shipping round for the v2 gate line (phase-aware gate, vocab 50261, blended
# pretrain). Single entrypoint for: pretrain -> probes -> sft -> smoke -> export
# on GCP, and ship (pull+verify+push) on the RTX4090.
#
# Freshness: cursors are auto-derived from the previous round's checkpoint
# (per_source_docs) via scripts/v11_data_cursor.py, so no web/reasoning doc is
# reused. The trainer also auto-seeds skips from --resume checkpoints.
#
# Self-driving rounds: the round number and round-<cumulative_B>b-gate tag are
# derived from a state file (PT_CKPT_DIR/round_state.env), and defaults are tuned
# for the optimized stack (B=32, FUSED_CE=1). So a normal next round is just:
#
# Usage (GCP, tmux):
#   # Round 1 from scratch (2B blended pretrain):
#   SCRATCH=1 tmux new-session -d -s v11_round './scripts/run_v11_round.sh pretrain'
#
#   # Any next round (auto: resume prev best, fresh tokens, next tag):
#   tmux new-session -d -s v11_round './scripts/run_v11_round.sh pretrain'
#   ./scripts/run_v11_round.sh probes
#   ./scripts/run_v11_round.sh sft
#   ./scripts/run_v11_round.sh smoke
#   ./scripts/run_v11_round.sh export
#
#   # Preview what a step would run without executing:
#   ./scripts/run_v11_round.sh pretrain --dry
#
# Ship (RTX4090):
#   ./scripts/run_v11_round.sh ship            # tag from state; or ROUND_TAG=... override

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

STEP="${1:-help}"; shift || true

# --dry / -n : print the resolved plan + exact command without executing.
DRY="${DRY:-0}"
for _a in "$@"; do
  case "$_a" in --dry|-n|--dry-run) DRY=1 ;; esac
done

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET="${PRESET:-v11_e3_k3_chat}"
# Competitive-retrieval A/B (2026-07-07) LOST to control on WikiText — do NOT use
# v11_e3_k3_chat_compete for rounds. v11_e3_k3_chat (gate-aware) remains the winner.
TOKEN_BUDGET="${TOKEN_BUDGET:-2000000000}"
SCRATCH="${SCRATCH:-0}"

PT_CKPT_DIR="${PT_CKPT_DIR:-checkpoints_v11_e3_k3_chat_pretrain_v2}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-checkpoints_v11_sft_chat_smoltalk_v2}"

# ── Self-driving round/tag ────────────────────────────────────────────────
# A tiny state file records the last completed pretrain round + cumulative
# pretrain tokens, so you can just run `./scripts/run_v11_round.sh pretrain`
# and it advances to the next `round-<cumulative_B>b-gate` automatically.
# `pretrain` advances the round; post-pretrain steps (probes/sft/smoke/export/
# eval) operate on the *current* (last completed) round. Env vars ROUND /
# ROUND_TAG / TOKEN_BUDGET still override anything below.
ROUND_STATE_FILE="${ROUND_STATE_FILE:-${PT_CKPT_DIR}/round_state.env}"
LAST_ROUND=0; CUMULATIVE_TOKENS=0; LAST_TAG=""
if [[ -f "$ROUND_STATE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ROUND_STATE_FILE"
fi
if [[ "$STEP" == "pretrain" ]]; then
  if [[ "$SCRATCH" == "1" ]]; then
    ROUND="${ROUND:-1}"
    CUM_AFTER="$TOKEN_BUDGET"
  else
    ROUND="${ROUND:-$((LAST_ROUND + 1))}"
    CUM_AFTER="$((CUMULATIVE_TOKENS + TOKEN_BUDGET))"
  fi
  CUM_B="$(( CUM_AFTER / 1000000000 ))"
  ROUND_TAG="${ROUND_TAG:-round-${CUM_B}b-gate}"
else
  ROUND="${ROUND:-${LAST_ROUND:-1}}"
  CUM_AFTER="$CUMULATIVE_TOKENS"
  ROUND_TAG="${ROUND_TAG:-${LAST_TAG:-round-${ROUND}}}"
fi

save_round_state() {
  [[ "$DRY" == "1" ]] && return 0
  mkdir -p "$(dirname "$ROUND_STATE_FILE")"
  cat > "$ROUND_STATE_FILE" <<EOF
LAST_ROUND=$ROUND
CUMULATIVE_TOKENS=$CUM_AFTER
LAST_TAG=$ROUND_TAG
EOF
  echo "[round-state] $ROUND_STATE_FILE -> round=$ROUND cumulative=${CUM_AFTER} tag=$ROUND_TAG"
}

dry_banner() {
  cat <<EOF
────────────────────────── DRY-RUN ──────────────────────────
 step            : $STEP
 round / tag     : $ROUND / $ROUND_TAG
 cumulative tok  : before=$CUMULATIVE_TOKENS  after=$CUM_AFTER
 token_budget    : $TOKEN_BUDGET   (this round)
 batch / seq     : $BATCH_SIZE / $SEQ_LEN   chunk=$CHUNK_SIZE
 lr / warmup     : pretrain=$LR/$WARMUP_STEPS   sft=${SFT_LR:-5e-5}/200
 compile / fused : COMPILE=$COMPILE  FUSED_CE=$FUSED_CE
 pt_ckpt_dir     : $PT_CKPT_DIR
 sft_ckpt_dir    : $SFT_CKPT_DIR
 state file      : $ROUND_STATE_FILE
──────────────────────────────────────────────────────────────
EOF
}

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
# Default B=32: fits in ~82GB with fused E3 + chunked CE (was 18 pre-optimization).
BATCH_SIZE="${BATCH_SIZE:-32}"
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

# Memory-lean chunked linear+CE (exact math): frees ~30GB at B=18/T=2048 so a
# larger BATCH_SIZE fits. Fused E3 (the 1.62x speedup) is always on by default in
# the model. On by default now (needed for B=32); set FUSED_CE=0 to disable.
FUSED_CE="${FUSED_CE:-1}"
FUSED_CE_ARGS=""
[[ "$FUSED_CE" == "1" ]] && FUSED_CE_ARGS="--fused_ce --fused_ce_chunk ${FUSED_CE_CHUNK:-4096}"

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

# CPU background token cache for the *next* pretrain round (~30GB, no GPU).
PREFETCH="${PREFETCH:-1}"
PARALLEL_PREFETCH="${PARALLEL_PREFETCH:-1}"
PREFETCH_MIN_DISK_GB="${PREFETCH_MIN_DISK_GB:-40}"

launch_prefetch() {
  local ckpt="$1"
  local offset="${2:-0}"
  local tag_round=$((ROUND + 1))
  local log="${PT_CKPT_DIR}/prefetch_r${tag_round}_off${offset}.log"
  if [[ "$PREFETCH" == "0" ]]; then
    echo "[prefetch] disabled (PREFETCH=0)"
    return 0
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "[prefetch] skip: checkpoint missing ($ckpt)"
    return 0
  fi
  local free_gb
  free_gb="$(df -BG / | awk 'NR==2{gsub("G","",$4); print $4}')"
  if (( free_gb < PREFETCH_MIN_DISK_GB )); then
    echo "[prefetch] skip: need >=${PREFETCH_MIN_DISK_GB}GB free (have ${free_gb}GB)"
    return 0
  fi
  if pgrep -f "v11_pretrain_prefetch.py.*--checkpoint ${ckpt}" >/dev/null 2>&1; then
    echo "[prefetch] already running for $ckpt"
    return 0
  fi
  echo "[prefetch] background build for round $tag_round (offset=${offset}) ckpt=$ckpt"
  nohup nice -n 10 env CUDA_VISIBLE_DEVICES="" \
    "$PYTHON_BIN" scripts/v11_pretrain_prefetch.py \
    --checkpoint "$ckpt" \
    --offset_tokens "$offset" \
    --token_budget "$TOKEN_BUDGET" \
    --seq_len "$SEQ_LEN" \
    --edu_score_min "$EDU_SCORE_MIN" \
    --pretrain_sources "$PRETRAIN_SOURCES" \
    --pretrain_weights "$PRETRAIN_WEIGHTS" \
    --fineweb_name "$FINEWEB_NAME" \
    --blend_warmup_tokens "$BLEND_WARMUP_TOKENS" \
    --mix_seed "$SEED" \
    >>"$log" 2>&1 &
  echo "[prefetch] pid=$! log=$log"
}

case "$STEP" in
prefetch)
  # Build token cache for the next pretrain round. Env:
  #   PREFETCH_CHECKPOINT  (default: pt_best)
  #   PREFETCH_OFFSET      (default: 0; use TOKEN_BUDGET for parallel prefetch)
  CKPT="${PREFETCH_CHECKPOINT:-$(pt_best)}"
  OFFSET="${PREFETCH_OFFSET:-0}"
  if [[ "$DRY" == "1" ]]; then
    dry_banner
    echo "[DRY] prefetch checkpoint=$CKPT offset=$OFFSET budget=$TOKEN_BUDGET"
    eval "$PYTHON_BIN scripts/v11_pretrain_prefetch.py" \
      --checkpoint "$CKPT" --offset_tokens "$OFFSET" \
      --token_budget "$TOKEN_BUDGET" --seq_len "$SEQ_LEN" \
      --edu_score_min "$EDU_SCORE_MIN" \
      --pretrain_sources "$PRETRAIN_SOURCES" \
      --pretrain_weights "$PRETRAIN_WEIGHTS" \
      --fineweb_name "$FINEWEB_NAME" \
      --blend_warmup_tokens "$BLEND_WARMUP_TOKENS" \
      --mix_seed "$SEED" --dry
    exit 0
  fi
  disk_check
  CUDA_VISIBLE_DEVICES="" eval "$PYTHON_BIN scripts/v11_pretrain_prefetch.py" \
    --checkpoint "$CKPT" --offset_tokens "$OFFSET" \
    --token_budget "$TOKEN_BUDGET" --seq_len "$SEQ_LEN" \
    --edu_score_min "$EDU_SCORE_MIN" \
    --pretrain_sources "$PRETRAIN_SOURCES" \
    --pretrain_weights "$PRETRAIN_WEIGHTS" \
    --fineweb_name "$FINEWEB_NAME" \
    --blend_warmup_tokens "$BLEND_WARMUP_TOKENS" \
    --mix_seed "$SEED"
  ;;

pretrain)
  disk_check
  mkdir -p "$PT_CKPT_DIR"
  RESUME_ARGS=""
  CURSOR_ARGS=""
  if [[ "$SCRATCH" == "1" ]]; then
    echo "[pretrain] from scratch (round $ROUND) -> $PT_CKPT_DIR"
  else
    # New round = weights-only resume (--resume_from): fresh warmup->cosine over
    # this round's TOKEN_BUDGET, fresh optimizer, and global_tokens reset to 0 so
    # the budget counts THIS round's tokens. Cursor seeding still uses the ckpt's
    # per_source_docs (fresh tokens, no reuse). Full-state resume (--resume) keeps
    # the old decayed schedule + cumulative counters and is ONLY for crash recovery
    # inside a round; enable with RESUME_FULL=1.
    if [[ "${RESUME_FULL:-0}" == "1" ]]; then
      RESUME_ARGS="--resume $(pt_best)"
      echo "[pretrain] FULL resume (crash recovery): schedule/optimizer/counters restored"
    else
      RESUME_ARGS="--resume_from $(pt_best)"
    fi
    # Auto cursors from the previous round's checkpoint (fresh tokens).
    eval "$(eval "$PYTHON_BIN scripts/v11_data_cursor.py --checkpoint $(pt_best) --fineweb-name $FINEWEB_NAME --emit env")"
    CURSOR_ARGS="--dclm_skip_docs ${DCLM_SKIP_DOCS:-0} --fineweb_skip_docs ${FINEWEB_SKIP_DOCS:-0} --smoltalk2_mid_skip_rows ${SMOLTALK2_MID_SKIP_ROWS:-0}"
    FINEWEB_NAME="${FINEWEB_NAME:-sample-10BT}"
    echo "[pretrain] resume $(pt_best) ($RESUME_ARGS) cursors: $CURSOR_ARGS fineweb=$FINEWEB_NAME"
  fi
  GEN_PROMPT="In 1923 , the University of"
  ARGS="--preset $PRESET --stage pretrain --dataset pretrain_mix --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE --epochs 9999 --chunk_size $CHUNK_SIZE \
    --token_budget $TOKEN_BUDGET --edu_score_min $EDU_SCORE_MIN \
    --pretrain_sources $PRETRAIN_SOURCES --pretrain_weights $PRETRAIN_WEIGHTS \
    --fineweb_name $FINEWEB_NAME --blend_warmup_tokens $BLEND_WARMUP_TOKENS \
    --seed $SEED --lr $LR --warmup_steps $WARMUP_STEPS \
    --amp_dtype auto --num_workers 0 --gen_every 5000 --save_every_steps $SAVE_EVERY_STEPS \
    --no_grad_ckpt $COMPILE_ARGS $FUSED_CE_ARGS $RESUME_ARGS $CURSOR_ARGS"
  if [[ "$DRY" == "1" ]]; then
    dry_banner
    echo "[DRY] resume    : ${RESUME_ARGS:-<scratch>}"
    echo "[DRY] cursors   : ${CURSOR_ARGS:-<none>}"
    echo "[DRY] on success would write state -> round=$ROUND cumulative=$CUM_AFTER tag=$ROUND_TAG"
    echo "[DRY] would run :"
    echo "  $PYTHON_BIN -m v11.train $ARGS --gen_prompt '$GEN_PROMPT' --log_dir <log> --checkpoint_dir $PT_CKPT_DIR"
    exit 0
  fi
  LOG_DIR=$(make_log_dir "v11" "round${ROUND}_pretrain")
  write_run_info "$LOG_DIR" "V11 round $ROUND pretrain ($ROUND_TAG)" "$ARGS"
  CURSOR_BASE="${PT_CKPT_DIR}/round${ROUND}_cursor_base.pt"
  if [[ "$SCRATCH" != "1" && -f "$(pt_best)" ]]; then
    cp -f "$(pt_best)" "$CURSOR_BASE"
    echo "[pretrain] saved cursor base $CURSOR_BASE (for parallel prefetch)"
    if [[ "$PARALLEL_PREFETCH" == "1" ]]; then
      launch_prefetch "$CURSOR_BASE" "$TOKEN_BUDGET"
    fi
  fi
  eval "$PYTHON_BIN -m v11.train" $ARGS \
    --gen_prompt "'$GEN_PROMPT'" --log_dir "$LOG_DIR" --checkpoint_dir "$PT_CKPT_DIR"
  save_round_state
  # Post-round prefetch (offset=0) if parallel did not finish / was disabled.
  if [[ "$SCRATCH" != "1" ]]; then
    launch_prefetch "$(pt_best)" 0
  fi
  ;;

probes)
  echo "[probes] gate + rank on $(pt_best)"
  if [[ "$DRY" == "1" ]]; then dry_banner; echo "[DRY] probes on $(pt_best)"; exit 0; fi
  eval "$PYTHON_BIN scripts/v11_probe_gates.py --checkpoint $(pt_best) --preset $PRESET" || true
  eval "$PYTHON_BIN -m memory_probes --test rank-text --checkpoint $(pt_best) --preset $PRESET --layer 4 --text-tokens 5000 --sample-every 100" || true
  ;;

sft)
  disk_check
  mkdir -p "$SFT_CKPT_DIR"
  GEN_PROMPT="<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
  ARGS="--preset $PRESET --stage sft --dataset smoltalk2 --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE --epochs ${SFT_EPOCHS:-1} --chunk_size $CHUNK_SIZE \
    --lr ${SFT_LR:-5e-5} --warmup_steps 200 --sft_filter hard \
    --think_fraction $THINK_FRACTION --smoltalk2_skip_rows ${SMOLTALK2_SKIP_ROWS:-0} \
    ${SFT_MAX_SAMPLES:+--max_samples $SFT_MAX_SAMPLES} \
    --amp_dtype auto --num_workers 4 --gen_every 0 --no_grad_ckpt $COMPILE_ARGS $FUSED_CE_ARGS \
    --resume_from $(pt_best)"
  if [[ "$DRY" == "1" ]]; then
    dry_banner
    echo "[DRY] resume_from: $(pt_best)"
    echo "[DRY] would run  :"
    echo "  $PYTHON_BIN -m v11.train $ARGS --gen_prompt '$GEN_PROMPT' --log_dir <log> --checkpoint_dir $SFT_CKPT_DIR"
    exit 0
  fi
  LOG_DIR=$(make_log_dir "v11" "round${ROUND}_sft")
  write_run_info "$LOG_DIR" "V11 round $ROUND smoltalk2 SFT ($ROUND_TAG)" "$ARGS"
  eval "$PYTHON_BIN -m v11.train" $ARGS \
    --gen_prompt "'$GEN_PROMPT'" --log_dir "$LOG_DIR" --checkpoint_dir "$SFT_CKPT_DIR"
  ;;

smoke)
  echo "[smoke] $(sft_best)"
  if [[ "$DRY" == "1" ]]; then dry_banner; echo "[DRY] smoke chat on $(sft_best) label=$ROUND_TAG"; exit 0; fi
  eval "$PYTHON_BIN scripts/smoke_chat_v11.py --checkpoint $(sft_best) --preset $PRESET --label $ROUND_TAG"
  ;;

export)
  disk_check
  echo "[export] $(sft_best) -> hf_release + server_manifest ($ROUND_TAG)"
  if [[ "$DRY" == "1" ]]; then
    dry_banner
    echo "[DRY] export $(sft_best) tag=$ROUND_TAG pretrain_tokens_total=${PRETRAIN_TOKENS_TOTAL:-$CUM_AFTER} round_tokens=$TOKEN_BUDGET"
    exit 0
  fi
  eval "$PYTHON_BIN scripts/export_hf_release.py \
    --src $(sft_best) --round $ROUND_TAG --tag $ROUND_TAG \
    --pretrain_tokens_total ${PRETRAIN_TOKENS_TOTAL:-$CUM_AFTER} \
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
  if [[ "$DRY" == "1" ]]; then dry_banner; echo "[DRY] eval_chat -> SAMPLES_${tag}.md"; exit 0; fi
  ( cd hf_release && eval "$PYTHON_BIN eval_chat.py" \
    --checkpoint qllm_v11_e3k3_chat.pt \
    --prompts eval_prompts_round1.yaml \
    --round-tag "$tag" \
    --out-md "SAMPLES_${tag}.md" \
    --out-json "../logs/v11/${tag}_chat_eval.json" )
  ;;

ship)
  echo "[ship] pull -> verify -> push ($ROUND_TAG) [run on RTX4090 with hf auth]"
  if [[ "$DRY" == "1" ]]; then dry_banner; echo "[DRY] pull+verify+push revision $ROUND_TAG"; exit 0; fi
  ROUND="$ROUND_TAG" ./scripts/pull_v11_release.sh --round "$ROUND_TAG"
  cp -f "releases/$ROUND_TAG/qllm_v11_e3k3_chat.pt" hf_release/qllm_v11_e3k3_chat.pt
  ( cd hf_release && bash verify.sh )
  eval "$PYTHON_BIN scripts/push_qllm_hf.py --revision $ROUND_TAG --also-main"
  echo "[ship] published revision $ROUND_TAG"
  ;;

help|*)
  cat <<EOF
run_v11_round.sh <step> [--dry]

  Self-driving: just run the step; round number + round-<cumulative_B>b-gate tag
  are derived from ${ROUND_STATE_FILE}. 'pretrain' advances the round and, on
  success, records the new cumulative state; probes/sft/smoke/export/eval act on
  the current (last completed) round.

  steps (GCP):  pretrain | prefetch | probes | sft | smoke | export | eval
  step (4090):  ship
  --dry, -n  :  print the resolved plan + exact command, run nothing.

  defaults:     BATCH_SIZE=$BATCH_SIZE  FUSED_CE=$FUSED_CE  TOKEN_BUDGET=$TOKEN_BUDGET
                PREFETCH=1  PARALLEL_PREFETCH=1 (CPU cache for next round's 2B)
  next round:   ROUND=$((LAST_ROUND + 1))  TAG=round-$(( (CUMULATIVE_TOKENS + TOKEN_BUDGET) / 1000000000 ))b-gate
  overrides:    ROUND ROUND_TAG SCRATCH TOKEN_BUDGET BATCH_SIZE FUSED_CE
                PREFETCH=0 PARALLEL_PREFETCH=0 RESUME_FULL(crash-recovery)
                PRETRAIN_WEIGHTS BLEND_WARMUP_TOKENS THINK_FRACTION FINEWEB_NAME
  typical:      ./scripts/run_v11_round.sh pretrain          # next round, auto tag
                ./scripts/run_v11_round.sh pretrain --dry    # preview only
                ./scripts/run_v11_round.sh prefetch --dry    # preview next-round cache
                ./scripts/run_v11_round.sh sft && ./scripts/run_v11_round.sh export
EOF
  ;;
esac
