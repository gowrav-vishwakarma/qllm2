#!/usr/bin/env bash
# A/B watchdog for the sempty complex-vs-real run (log format: "step N  loss=X  lr=Y").
# Usage: bash tmp_ab_watchdog.sh <log-dir> <verdict_step> <max_seconds>
# Wakes on: train process gone / error in logs / verdict step reached / timeout.
set -u
LOGDIR="${1:?log dir}"
VERDICT_STEP="${2:-4300}"
MAX_S="${3:-2940}"
cd /home/gowrav/Development/qllm2

END=$(( $(date +%s) + MAX_S ))
REASON="timeout"
while [ $(date +%s) -lt $END ]; do
    # liveness: a python process running v13_sempty.train, or the driver session alive
    live=0
    for pid in $(pgrep -f "v13_sempty\.train" 2>/dev/null); do
        exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || echo "")
        case "$exe" in *python*) live=1; break;; esac
    done
    [ "$live" -eq 0 ] && tmux has-session -t sempty_ab 2>/dev/null && live=1
    if [ "$live" -eq 0 ]; then REASON="TRAIN-PROCESS-GONE"; break; fi

    grep -qE "OutOfMemoryError|CUDA out of memory|Traceback \(most recent" "$LOGDIR"/ab_*.log 2>/dev/null && { REASON="TRAIN-ERROR-IN-LOG"; break; }

    # max step across both arms' logs
    STEP=$(grep -hoE "^step [0-9]+" "$LOGDIR"/ab_*.log 2>/dev/null | awk '{print $2}' | sort -n | tail -1)
    [ -n "$STEP" ] && [ "$STEP" -ge "$VERDICT_STEP" ] && { REASON="VERDICT-POINT-REACHED"; break; }

    sleep 60
done
echo "WATCHDOG EXIT: $REASON  (epoch=$(date +%s))"
echo "--- latest 16 sampled step lines (both arms) ---"
grep -hE "^step " "$LOGDIR"/ab_*.log 2>/dev/null | awk 'NR==1 || NR%25==0' | tail -16
tmux ls 2>/dev/null | grep -E "sempty" || echo "no sempty tmux sessions"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
