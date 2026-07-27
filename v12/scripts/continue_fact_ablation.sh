#!/usr/bin/env bash
# Follow-on pipeline for the V12 fact ablation — does NOT touch a live training run.
#
# Waits for the current delta fact_retrieval job (tmux session / PID) to finish,
# then chains the next ablation steps while keeping the delta module intact:
#   1. Salvage compact+publish if train ended without them
#   2. Compose + recall-eval the delta arm (fact_retrieval@1.0)
#   3. Train the additive A/B arm as fact_retrieval@1.1 (separate ckpt root)
#   4. Compose + recall-eval the additive arm
#   5. (optional) Grow reasoning on the pinned delta substrate
#
# Designed to be launched in a *separate* tmux session while delta is still
# training, e.g.:
#   tmux new-session -d -s v12fact_followon \
#     'v12/scripts/continue_fact_ablation.sh 2>&1 | tee logs/v12_smoke_fact/followon.log'
#
# Env:
#   WAIT_TMUX (default v12fact)  — session whose shell must exit before we start
#   WAIT_PID                     — optional explicit PID to wait on (overrides tmux)
#   BATCH / SEQ / TOKEN_BUDGET / LR / GEN_EVERY — match the live delta run
#   RUN_REASONING=1|0 (default 1)
#   TOKEN_BUDGET_REASONING (default = TOKEN_BUDGET)
#   RECALL_TRIALS (default 30)
#   START_FROM=additive|reasoning — skip already-finished stages (resume helper)
#   SKIP_DELTA_EVAL / SKIP_ADDITIVE — fine-grained skips (also set by START_FROM)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

PY="${PY:-.venv/bin/python}"
REGISTRY="${REGISTRY:-v12_registry}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints_v12_curriculum}"
LOGDIR="${LOGDIR:-logs/v12_smoke_fact}"
WAIT_TMUX="${WAIT_TMUX:-v12fact}"
BATCH="${BATCH:-32}"
SEQ="${SEQ:-1024}"
TOKEN_BUDGET="${TOKEN_BUDGET:-300000000}"
TOKEN_BUDGET_REASONING="${TOKEN_BUDGET_REASONING:-$TOKEN_BUDGET}"
LR="${LR:-}"                    # empty => train_curriculum fact uses FACT_LR=3e-5
FACT_LR="${FACT_LR:-3e-5}"
GEN_EVERY="${GEN_EVERY:-5000}"
SAVE_EVERY_STEPS_FACT="${SAVE_EVERY_STEPS_FACT:-1000}"
RUN_REASONING="${RUN_REASONING:-1}"
RECALL_TRIALS="${RECALL_TRIALS:-30}"
VER_DELTA="${VER_DELTA:-1.0}"
VER_ADDITIVE="${VER_ADDITIVE:-1.1}"
CKPT_ADDITIVE="${CKPT_ADDITIVE:-checkpoints_v12_curriculum_additive}"
# Skip stages already finished (resume after a partial follow-on).
SKIP_DELTA_EVAL="${SKIP_DELTA_EVAL:-0}"   # 1 = skip compose+eval of delta@1.0
SKIP_ADDITIVE="${SKIP_ADDITIVE:-0}"       # 1 = skip additive train+eval
START_FROM="${START_FROM:-}"              # additive|reasoning — shorthand skip flags

mkdir -p "$LOGDIR" packed_v12
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" | tee -a "$LOGDIR/followon.log"; }

# True iff a *real* delta fact job is still alive (excludes Cursor sandbox leftovers
# and this follow-on script itself).
live_fact_job_running() {
  ps -eo pid,cmd | awk '
    /cursorsandbox|continue_fact_ablation|awk/ { next }
    /bash .*train_curriculum\.sh fact_retrieval/ { found=1 }
    /python .*v12\.train .*--dataset fact/ { found=1 }
    END { exit found ? 0 : 1 }
  '
}

# ── 0. Wait for the live delta job (do not kill / restart it) ─────────────────
wait_for_live() {
  if [ -n "${WAIT_PID:-}" ]; then
    # Refuse to wait on a sandbox leftover (that was the earlier footgun).
    if ps -p "$WAIT_PID" -o cmd= 2>/dev/null | grep -q cursorsandbox; then
      log "ERROR: WAIT_PID=$WAIT_PID is a cursorsandbox leftover — refusing to wait on it"
      log "falling back to live process detection"
      unset WAIT_PID
    fi
  fi
  if [ -n "${WAIT_PID:-}" ]; then
    if ! kill -0 "$WAIT_PID" 2>/dev/null; then
      log "WAIT_PID=$WAIT_PID already gone"
    else
      cmd=$(ps -p "$WAIT_PID" -o cmd= 2>/dev/null || true)
      log "waiting for PID $WAIT_PID to exit ($cmd)"
      while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
      log "PID $WAIT_PID exited"
    fi
    # Also wait out compact/publish if a sibling wrapper is still finishing.
    while live_fact_job_running; do sleep 30; done
    return
  fi
  if live_fact_job_running || tmux has-session -t "$WAIT_TMUX" 2>/dev/null; then
    log "waiting for live delta fact job / tmux '$WAIT_TMUX' to finish..."
    while live_fact_job_running; do sleep 60; done
    # Session may linger at a shell prompt after the one-shot command; that's fine.
    log "live delta wait complete"
    return
  fi
  log "no live delta job detected — continuing from published/salvaged state"
}

salvage_delta_publish() {
  local best="$CKPT_ROOT/fact_retrieval/best_model.pt"
  local slim="$CKPT_ROOT/fact_retrieval/slim.pt"
  local reg_pt="$REGISTRY/fact_retrieval/${VER_DELTA}/model.pt"
  if [ -f "$reg_pt" ]; then
    log "delta already published at $reg_pt"
    return 0
  fi
  if [ -f "$best" ] && [ ! -f "$slim" ]; then
    log "salvaging compact after train (best exists, slim missing)"
    "$PY" -m v12.compact --checkpoint "$best" --out "$slim" --threshold 1e-3 \
      2>&1 | tee "$LOGDIR/02_compact_salvage.log"
  fi
  if [ -f "$slim" ] || [ -f "$best" ]; then
    local src="${slim}"
    [ -f "$src" ] || src="$best"
    log "salvaging publish fact_retrieval@${VER_DELTA} from $src"
    "$PY" -m v12.publish --checkpoint "$src" \
      --module_id fact_retrieval --version "$VER_DELTA" --role group \
      --group_id fact_retrieval \
      --requires "grammar@>=${VER_DELTA}:prelayer" \
      --provenance local --registry "$REGISTRY" --overwrite \
      2>&1 | tee "$LOGDIR/02_publish_salvage.log"
  else
    log "ERROR: no best_model/slim/published delta — cannot continue"
    return 1
  fi
}

compose_eval() {
  local tag="$1" constraint="$2" out="$3"
  local rc=0
  log "=== compose+eval ($tag, constraint=$constraint) ==="
  CONSTRAINT="$constraint" OUT="$out" \
    bash v12/scripts/compose.sh fact_retrieval \
    2>&1 | tee "$LOGDIR/03_compose_${tag}.log"
  # PPL path can fail (missing holdout cache / network); never abort the ablation.
  set +e
  RECALL=1 RECALL_TRIALS="$RECALL_TRIALS" \
    bash v12/scripts/eval.sh "$out" \
    2>&1 | tee "$LOGDIR/03_eval_${tag}.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    log "WARN: eval.sh exit=$rc for $tag (continuing; dedicated recall still runs)"
  fi
  # Dedicated recall JSON — the fact scorecard.
  "$PY" -m v12.eval_recall --checkpoint "$out" \
    --context-lengths 128,512,1024,2048 \
    --association-counts 1,4,8 \
    --trials "$RECALL_TRIALS" \
    --output "$LOGDIR/recall_${tag}.json" \
    2>&1 | tee -a "$LOGDIR/03_eval_${tag}.log"
  log "=== compose+eval ($tag) DONE ==="
}

# ── main ──────────────────────────────────────────────────────────────────────
case "$START_FROM" in
  additive)  SKIP_DELTA_EVAL=1 ;;
  reasoning) SKIP_DELTA_EVAL=1; SKIP_ADDITIVE=1 ;;
  ""|delta)  ;;
  *) log "WARN: unknown START_FROM=$START_FROM (ignored)" ;;
esac

log "=== follow-on pipeline START (will not touch the live delta run) ==="
log "wait_tmux=$WAIT_TMUX batch=$BATCH budget=$TOKEN_BUDGET reasoning=$RUN_REASONING"
log "skip: delta_eval=$SKIP_DELTA_EVAL additive=$SKIP_ADDITIVE start_from=${START_FROM:-none}"
wait_for_live

log "=== STAGE A: finalize delta publish ==="
salvage_delta_publish

if [ "$SKIP_DELTA_EVAL" = "1" ]; then
  log "SKIP STAGE B (delta compose+eval) — already done or START_FROM=$START_FROM"
else
  log "=== STAGE B: delta compose + recall eval ==="
  compose_eval delta "$VER_DELTA" "packed_v12/delta.pt"
fi

if [ "$SKIP_ADDITIVE" = "1" ]; then
  log "SKIP STAGE C/D (additive arm)"
else
  log "=== STAGE C: additive A/B fact arm (fact_retrieval@${VER_ADDITIVE}) ==="
  # VER=1.1 is the *module* version we publish; substrate must stay grammar@1.0
  # (pinning VER=1.1 would wrongly require grammar@>=1.1).
  FACT_MODE=additive \
  VER="$VER_ADDITIVE" \
  SUBSTRATE="grammar@>=${VER_DELTA}" \
  REQUIRES="grammar@>=${VER_DELTA}:prelayer" \
  CKPT_ROOT="$CKPT_ADDITIVE" \
  BATCH="$BATCH" SEQ="$SEQ" TOKEN_BUDGET="$TOKEN_BUDGET" \
  FACT_LR="$FACT_LR" \
  GEN_EVERY="$GEN_EVERY" SAVE_EVERY_STEPS_FACT="$SAVE_EVERY_STEPS_FACT" \
  REGISTRY="$REGISTRY" PY="$PY" \
    bash v12/scripts/train_curriculum.sh fact_retrieval \
    2>&1 | tee "$LOGDIR/04_fact_additive.log"
  log "=== STAGE C DONE ==="

  log "=== STAGE D: additive compose + recall eval ==="
  compose_eval additive "$VER_ADDITIVE" "packed_v12/additive.pt"
fi

if [ "$RUN_REASONING" = "1" ]; then
  log "=== STAGE E: reasoning on pinned delta substrate (fact@${VER_DELTA}) ==="
  # Pin delta so the later additive@1.1 is NOT pulled in as substrate.
  SUBSTRATE="grammar@>=${VER_DELTA},fact_retrieval@${VER_DELTA}" \
  REQUIRES="grammar@>=${VER_DELTA}:prelayer,fact_retrieval@${VER_DELTA}:prelayer" \
  VER="$VER_DELTA" \
  CKPT_ROOT="$CKPT_ROOT" \
  BATCH="$BATCH" SEQ="$SEQ" TOKEN_BUDGET="$TOKEN_BUDGET_REASONING" \
  LR="${REASONING_LR:-1e-4}" \
  GEN_EVERY="$GEN_EVERY" \
  REGISTRY="$REGISTRY" PY="$PY" \
    bash v12/scripts/train_curriculum.sh reasoning \
    2>&1 | tee "$LOGDIR/05_reasoning.log"
  CONSTRAINT="$VER_DELTA" OUT="packed_v12/reasoning_on_delta.pt" \
    bash v12/scripts/compose.sh reasoning \
    2>&1 | tee "$LOGDIR/05_compose_reasoning.log"
  set +e
  RECALL=1 RECALL_TRIALS="$RECALL_TRIALS" \
    bash v12/scripts/eval.sh packed_v12/reasoning_on_delta.pt \
    2>&1 | tee "$LOGDIR/05_eval_reasoning.log"
  set -e
  log "=== STAGE E DONE ==="
else
  log "SKIP reasoning (RUN_REASONING=$RUN_REASONING)"
fi

log "=== follow-on pipeline COMPLETE ==="
log "artifacts: packed_v12/delta.pt packed_v12/additive.pt $LOGDIR/recall_*.json"
