#!/usr/bin/env bash
# Backup the server pretrain checkpoint on the RTX4090 for disaster recovery.
#
# Pulled alongside the HF SFT export (pull_v11_release.sh / ship). Keeps only what
# is needed to resume the next pretrain round on a new GCP host:
#   checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt
#   checkpoints_v11_e3_k3_chat_pretrain_v2/round_state.env
#
# After verify + HF push, run --cleanup to drop redundant local copies (latest.pt,
# round cursor snapshots, raw SFT weights) and reclaim ~3–4 GB.
#
# Usage:
#   ./scripts/pull_v11_training_ckpt.sh
#   ./scripts/pull_v11_training_ckpt.sh --cleanup
#   ./scripts/pull_v11_training_ckpt.sh --dry-run
#
# Env overrides:
#   REMOTE  REMOTE_DIR  SSH_KEY
#   PT_CKPT_DIR  LOCAL_PT_CKPT_DIR

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

REMOTE="${REMOTE:-ubuntu@34.131.232.173}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/Development/qllm-private}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/gowrav-personal}"
PT_CKPT_DIR="${PT_CKPT_DIR:-checkpoints_v11_e3_k3_chat_pretrain_v2}"
LOCAL_PT_CKPT_DIR="${LOCAL_PT_CKPT_DIR:-$(pwd)/$PT_CKPT_DIR}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-checkpoints_v11_sft_chat_smoltalk_v2}"
LOCAL_SFT_CKPT_DIR="${LOCAL_SFT_CKPT_DIR:-$(pwd)/$SFT_CKPT_DIR}"

MODE="pull"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cleanup) MODE="cleanup" ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

SSH_OPTS=(-o BatchMode=yes)
[[ -f "$SSH_KEY" ]] && SSH_OPTS+=(-i "$SSH_KEY")

pull_one() {
  local remote_rel="$1"
  local local_path="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] would pull ${REMOTE}:${REMOTE_DIR}/${remote_rel} -> ${local_path}"
    return 0
  fi
  mkdir -p "$(dirname "$local_path")"
  echo "[training-pull] ${remote_rel}"
  scp "${SSH_OPTS[@]}" "${REMOTE}:${REMOTE_DIR}/${remote_rel}" "${local_path}.partial"
  mv -f "${local_path}.partial" "${local_path}"
  ls -lh "$local_path"
}

do_pull() {
  echo "[training-pull] backing up pretrain checkpoint for DR"
  pull_one "${PT_CKPT_DIR}/round_state.env" "${LOCAL_PT_CKPT_DIR}/round_state.env"
  pull_one "${PT_CKPT_DIR}/best_model.pt" "${LOCAL_PT_CKPT_DIR}/best_model.pt"
  echo "[training-pull] done -> ${LOCAL_PT_CKPT_DIR}"
}

do_cleanup() {
  echo "[training-pull] cleanup: keep best_model.pt + round_state.env only"
  local removed=0
  cleanup_file() {
    local path="$1"
    [[ -e "$path" ]] || return 0
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run] would remove ${path}"
      return 0
    fi
    rm -f "$path"
    echo "[training-pull] removed ${path}"
    removed=1
  }

  cleanup_file "${LOCAL_PT_CKPT_DIR}/latest.pt"
  cleanup_file "${LOCAL_PT_CKPT_DIR}/final_model.pt"
  for f in "${LOCAL_PT_CKPT_DIR}"/round*_cursor_base.pt "${LOCAL_PT_CKPT_DIR}"/round-*_base.pt; do
    [[ -e "$f" ]] || continue
    cleanup_file "$f"
  done
  cleanup_file "${LOCAL_SFT_CKPT_DIR}/best_model.pt"
  cleanup_file "${LOCAL_SFT_CKPT_DIR}/final_model.pt"
  cleanup_file "${LOCAL_SFT_CKPT_DIR}/latest.pt"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] cleanup preview complete"
  elif [[ "$removed" == "0" ]]; then
    echo "[training-pull] nothing extra to remove"
  else
    echo "[training-pull] cleanup done; kept:"
    ls -lh "${LOCAL_PT_CKPT_DIR}/best_model.pt" "${LOCAL_PT_CKPT_DIR}/round_state.env" 2>/dev/null || true
  fi
}

case "$MODE" in
  pull) do_pull ;;
  cleanup) do_cleanup ;;
esac
