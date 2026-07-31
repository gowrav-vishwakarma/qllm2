#!/usr/bin/env bash
# Phase 0 (V12 next phase): 50-value control, pool sweep, full-strength contrastive.
# Matches attempt-C settings except FACT_VALUE_POOL and optional ce_fact_strong loss.
#
# Usage:
#   v12/scripts/run_phase0_controls.sh control50     # ~80 min on 4090
#   v12/scripts/run_phase0_controls.sh sweep           # 50, 200, 1000 sequential
#   v12/scripts/run_phase0_controls.sh contrastive     # fact_contrastive_lambda=0.5
#   v12/scripts/run_phase0_controls.sh grid control50  # transfer grid on checkpoint
#   v12/scripts/run_phase0_controls.sh remaining       # pool200 -> contrastive -> pool1000
#
# Env: SUBSTRATE=grammar@1.0 VER=3.0 REQ_VER=1.0 CKPT_ROOT=checkpoints_v12_phase0
#   RUN_STAMP — suffix for log paths (default: date + time); re-runs keep history.
#   Tee + v12.train logs: logs/v12_phase0/runs/ and .../internal/ (never logs/v12_*.log).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

PY="${PY:-.venv/bin/python}"
REGISTRY="${REGISTRY:-v12_registry}"
SUBSTRATE="${SUBSTRATE:-grammar@1.0}"
VER="${VER:-3.0}"
# Substrate version for --requires (must match the grammar module actually used).
# VER is the version of the *published fact module*; do not reuse it for requires.
REQ_VER="${REQ_VER:-1.0}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints_v12_phase0}"
LOG_ROOT="${LOG_ROOT:-logs/v12_phase0}"
GRID_DIR="${LOG_ROOT}/grid"
LOCK_FILE="${LOCK_FILE:-${LOG_ROOT}/phase0.lock}"

mkdir -p "$LOG_ROOT" "$GRID_DIR"

_root_train_pids() {
  # DataLoader workers inherit the same cmdline; only count roots (parent not also train).
  local p ppid
  for p in $(pgrep -f '[v]12[.]train' 2>/dev/null || true); do
    ppid="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
    if [ -n "$ppid" ] && ! ps -p "$ppid" -o cmd= 2>/dev/null | grep -q '[v]12[.]train'; then
      echo "$p"
    fi
  done
}

_guard_no_duplicate() {
  # Abort if another root v12.train is already on the GPU (duplicate launch = 6x slower).
  local roots
  roots="$(_root_train_pids)"
  if [ -n "$roots" ]; then
    echo "ERROR: another v12.train is already running (root pids):" >&2
    echo "$roots" | while read -r p; do ps -p "$p" -o pid=,cmd= 2>/dev/null; done >&2 || true
    echo "Refuse to stack trainers. Wait or kill the other job first." >&2
    exit 1
  fi
  if [ -f "$LOCK_FILE" ]; then
    local old_pid
    old_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [ -n "${old_pid:-}" ] && kill -0 "$old_pid" 2>/dev/null; then
      echo "ERROR: phase0 lock held by pid $old_pid ($LOCK_FILE)" >&2
      exit 1
    fi
    echo "Stale lock $LOCK_FILE (pid ${old_pid:-?}); removing."
    rm -f "$LOCK_FILE"
  fi
  echo $$ > "$LOCK_FILE"
  trap '_release_lock' EXIT INT TERM
}

_release_lock() {
  if [ -f "$LOCK_FILE" ] && [ "$(cat "$LOCK_FILE" 2>/dev/null)" = "$$" ]; then
    rm -f "$LOCK_FILE"
  fi
}

_attempt_c_env() {
  export BATCH=8 SEQ=512 TOKEN_BUDGET=40000000 FACT_LR=3e-5
  export GEN_EVERY=0 SAVE_EVERY_STEPS_FACT=1000
  export SUBSTRATE CKPT_ROOT REGISTRY VER
  export REQUIRES="grammar@>=${REQ_VER}:prelayer"
  export DATASET=fact STAGE_LOSS=ce_fact FACT_MODE=delta FACT_LAYERS=4
}

train_fact() {
  local tag="$1"
  local pool="$2"
  local freeze="${3:-0}"
  local loss="${4:-ce_fact}"
  _attempt_c_env
  export FACT_VALUE_POOL="$pool"
  export FREEZE_SHARED="$freeze"
  export STAGE_LOSS="$loss"
  # Split locals: bash expands ${tag} before assignment under set -u.
  local stamp log log_dir
  stamp="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
  log_dir="$LOG_ROOT/internal/${tag}_pool${pool}_${stamp}"
  log="$LOG_ROOT/runs/${tag}_pool${pool}_${stamp}.log"
  mkdir -p "$log_dir" "$(dirname "$log")"
  export LOG_DIR="$log_dir"
  echo "=== train $tag pool=$pool freeze=$freeze loss=$loss requires=$REQUIRES"
  echo "    tee=$log  v12.train LOG_DIR=$LOG_DIR"
  PYTHONUNBUFFERED=1 \
    CKPT_ROOT="$CKPT_ROOT/$tag" \
    VER="$VER" \
    REQUIRES="$REQUIRES" \
    SUBSTRATE="$SUBSTRATE" \
    v12/scripts/train_curriculum.sh fact_retrieval 2>&1 | tee "$log"
  # Convenience symlink to latest run for this arm (does not remove older runs).
  ln -sfn "$(basename "$log")" "$LOG_ROOT/${tag}_pool${pool}.log.latest"
}

run_grid() {
  local tag="$1"
  local pool="${2:-50}"
  local ckpt
  local out
  ckpt="$CKPT_ROOT/${tag}/fact_retrieval/best_model.pt"
  out="$GRID_DIR/${tag}.json"
  if [ ! -f "$ckpt" ]; then
    echo "ERROR: missing checkpoint $ckpt" >&2
    return 1
  fi
  echo "=== grid $tag pool=$pool -> $out"
  "$PY" -m v12.diagnose_fact_shortcut --grid --trials 200 \
    --checkpoint "$ckpt" --seq_len 512 --value_pool "$pool" \
    --output "$out"
}

# Run arm; log status; never abort the whole multi-arm sequence on a single fail.
_run_arm() {
  local name="$1"
  shift
  echo "######## ARM $name ########"
  if "$@"; then
    echo "######## ARM $name OK ########"
    return 0
  else
    local rc=$?
    echo "######## ARM $name FAILED (rc=$rc) ########" >&2
    return "$rc"
  fi
}

CMD="${1:-help}"

case "$CMD" in
  control50|control50_frozen|pool200|pool1000|contrastive|sweep|remaining)
    _guard_no_duplicate
    ;;
  grid)
    # Grid-only is read-only on GPU; still take the lock so we don't collide with a train.
    _guard_no_duplicate
    ;;
esac

case "$CMD" in
  control50)
    _run_arm control50 train_fact control50 50 0 ce_fact
    _run_arm control50_grid run_grid control50 50
    ;;
  control50_frozen)
    _run_arm control50_frozen train_fact control50_frozen 50 1 ce_fact
    _run_arm control50_frozen_grid run_grid control50_frozen 50
    ;;
  pool200)
    _run_arm pool200 train_fact pool200 200 0 ce_fact
    _run_arm pool200_grid run_grid pool200 200
    ;;
  pool1000)
    _run_arm pool1000 train_fact pool1000 1000 0 ce_fact
    _run_arm pool1000_grid run_grid pool1000 1000
    ;;
  contrastive)
    _run_arm contrastive train_fact contrastive 50 0 ce_fact_strong
    _run_arm contrastive_grid run_grid contrastive 50
    ;;
  sweep)
    # ; not && — one failure must not cancel later arms.
    status=0
    for p in 50 200 1000; do
      _run_arm "pool${p}" train_fact "pool${p}" "$p" 0 ce_fact || status=1
      _run_arm "pool${p}_grid" run_grid "pool${p}" "$p" || status=1
    done
    exit "$status"
    ;;
  remaining)
    # Complete the queue after control50 already finished.
    status=0
    _run_arm pool200 train_fact pool200 200 0 ce_fact || status=1
    _run_arm pool200_grid run_grid pool200 200 || status=1
    _run_arm contrastive train_fact contrastive 50 0 ce_fact_strong || status=1
    _run_arm contrastive_grid run_grid contrastive 50 || status=1
    _run_arm pool1000 train_fact pool1000 1000 0 ce_fact || status=1
    _run_arm pool1000_grid run_grid pool1000 1000 || status=1
    exit "$status"
    ;;
  grid)
    run_grid "${2:?tag}" "${3:-50}"
    ;;
  *)
    echo "Usage: $0 {control50|control50_frozen|pool200|pool1000|contrastive|sweep|remaining|grid TAG [pool]}"
    exit 1
    ;;
esac
