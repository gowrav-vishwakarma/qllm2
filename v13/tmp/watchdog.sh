#!/usr/bin/env bash
# V13 WATCHDOG — finite ~50min monitor. Exits EARLY on: training process death,
# OOM/traceback, or a verdict gtok threshold. Its auto-delivery wakes the agent,
# which MUST re-arm the next watchdog (chain). Usage:
#   bash v13/tmp/watchdog.sh <log_file> <verdict_gtok> [max_seconds]
# Prints a compact status snapshot on exit.
set -u
LOG="${1:?log file}"
VERDICT_GTok="${2:-52000000}"
MAX_S="${3:-2940}"
cd /home/gowrav/Development/qllm2
END=$(( $(date +%s) + MAX_S ))
REASON="timeout"
while [ $(date +%s) -lt $END ]; do
    if ! pgrep -f "v1[13].train" >/dev/null 2>&1; then REASON="TRAIN-PROCESS-GONE"; break; fi
    if grep -qE "OutOfMemoryError|CUDA out of memory|Traceback \(most recent" "$LOG" 2>/dev/null; then
        REASON="TRAIN-ERROR-IN-LOG"; break
    fi
    GTOK=$(grep -oE "gtok=[0-9]+" "$LOG" 2>/dev/null | tail -1 | cut -d= -f2)
    if [ -n "$GTOK" ] && [ "$GTOK" -ge "$VERDICT_GTok" ]; then REASON="VERDICT-POINT-REACHED"; break; fi
    sleep 60
done
echo "WATCHDOG EXIT: $REASON"
echo "--- last steps ---"
grep -oE "\[1\] [0-9]+ loss=[0-9.]+ .*gtok=[0-9]+" "$LOG" 2>/dev/null \
  | sed -E 's/\[1\] ([0-9]+) loss=([0-9.]+) ppl=([0-9.]+) lr=([0-9.e-]+) \| ([0-9]+) tok\/s.*gtok=([0-9]+)/step=\1 loss=\2 lr=\4 tok_s=\5 gtok=\6/' \
  | awk 'NR==1 || NR%25==0' | tail -12
grep -oE "\[1\] [0-9]+ loss=[0-9.]+ .*gtok=[0-9]+" "$LOG" 2>/dev/null \
  | sed -E 's/\[1\] ([0-9]+) loss=([0-9.]+) ppl=([0-9.]+) lr=([0-9.e-]+) \| ([0-9]+) tok\/s.*gtok=([0-9]+)/step=\1 loss=\2 lr=\4 tok_s=\5 gtok=\6/' \
  | tail -10
tmux ls 2>/dev/null | grep -E "v13|v11" || echo "no v13/v11 tmux sessions"
echo "--- gpu ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
