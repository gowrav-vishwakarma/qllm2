#!/usr/bin/env bash
# Monitor GPU and optionally tail V6 training log.
#
# Log files are auto-created by train.py at: logs/v6_train_{size}.log
# e.g. logs/v6_train_small-matched.log
#
# Usage:
#   ./scripts/monitor_v6.sh
#   ./scripts/monitor_v6.sh 10
#   ./scripts/monitor_v6.sh 5 logs/v6_train_small-matched.log

INTERVAL="${1:-5}"
LOG_FILE="${2:-}"

if [[ -n "$LOG_FILE" ]]; then
  echo "Watching GPU every ${INTERVAL}s and tailing $LOG_FILE"
  while true; do
    clear
    echo "=== $(date) ==="
    nvidia-smi 2>/dev/null || echo "(nvidia-smi not available)"
    echo ""
    echo "=== Last 30 lines of $LOG_FILE ==="
    tail -30 "$LOG_FILE" 2>/dev/null || echo "(file not found or empty)"
    sleep "$INTERVAL"
  done
else
  echo "Watching GPU every ${INTERVAL}s (Ctrl+C to stop)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    watch -n "$INTERVAL" nvidia-smi
  else
    echo "(nvidia-smi not available -- use 'top' or Activity Monitor on Mac)"
  fi
fi
