#!/usr/bin/env bash
# Pull latest.pt from the remote V11 scratch pretrain run.
#
# Usage:
#   ./scripts/pull_v11_checkpoint.sh              # one-shot pull
#   ./scripts/pull_v11_checkpoint.sh --watch 30m  # repeat every 30 minutes
#
# Cron example (every 2 hours, aligned with server save_every_steps=5000):
#   15 */2 * * * /home/gowrav/Development/qllm2/scripts/pull_v11_checkpoint.sh >> /home/gowrav/Development/qllm2/logs/pull_v11_checkpoint.log 2>&1

set -euo pipefail

REMOTE="${REMOTE:-ubuntu@34.131.232.173}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/Development/qllm-private/checkpoints_v11_e3_k3_chat_pretrain}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)/checkpoints_v11_e3_k3_chat_pretrain}"
FILE="${FILE:-latest.pt}"

WATCH_INTERVAL=""
if [[ "${1:-}" == "--watch" ]]; then
  WATCH_INTERVAL="${2:-2h}"
fi

pull_once() {
  mkdir -p "$LOCAL_DIR"
  local remote_path="${REMOTE}:${REMOTE_DIR}/${FILE}"
  local local_path="${LOCAL_DIR}/${FILE}"
  local tmp_path="${local_path}.partial"

  echo "[$(date -Is)] pulling ${remote_path} -> ${local_path}"

  scp -o BatchMode=yes "${remote_path}" "${tmp_path}"
  mv -f "${tmp_path}" "${local_path}"
  echo "[$(date -Is)] done: $(ls -lh "${local_path}")"
}

if [[ -n "$WATCH_INTERVAL" ]]; then
  echo "Watching every ${WATCH_INTERVAL} (Ctrl-C to stop)"
  while true; do
    pull_once || echo "[$(date -Is)] pull failed, will retry"
    sleep "${WATCH_INTERVAL}"
  done
else
  pull_once
fi
