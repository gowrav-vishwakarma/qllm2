#!/usr/bin/env bash
# Monitor GPU and optionally tail training log. Run in a separate terminal/tmux pane.
# Usage:
#   ./scripts/monitor_training.sh              # GPU every 5s
#   ./scripts/monitor_training.sh 10           # GPU every 10s
#   ./scripts/monitor_training.sh 5 logs/v4_medium_20250128.log  # GPU + tail log

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
