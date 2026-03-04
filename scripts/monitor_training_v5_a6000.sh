#!/usr/bin/env bash
# Monitor GPU and optionally tail v5 training log.
#
# Log files are auto-created by train.py at: logs/v5_train_{size}.log
# e.g. logs/v5_train_small-matched.log, logs/v5_train_small.log
#
# Usage:
#   ./scripts/monitor_training_v5_a6000.sh
#   ./scripts/monitor_training_v5_a6000.sh 10
#   ./scripts/monitor_training_v5_a6000.sh 5 logs/v5_train_small-matched.log

INTERVAL="${1:-5}"
LOG_FILE="${2:-}"

if [[ -n "$LOG_FILE" ]]; then
  echo "Watching GPU every ${INTERVAL}s and tailing $LOG_FILE"
  while true; do
    clear
    echo "=== $(date) ==="
    nvidia-smi
    echo ""
    echo "=== Last 30 lines of $LOG_FILE ==="
    tail -30 "$LOG_FILE" 2>/dev/null || echo "(file not found or empty)"
    sleep "$INTERVAL"
  done
else
  echo "Watching GPU every ${INTERVAL}s (Ctrl+C to stop)"
  watch -n "$INTERVAL" nvidia-smi
fi

