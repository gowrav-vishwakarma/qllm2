#!/usr/bin/env bash
# Watchdog for the sempty wikitext real run (log format: "step N [..]  loss=X").
# Usage: bash tmp_wiki_watchdog.sh <log-file> <verdict_step> <max_seconds>
# Wakes on: train process gone / error in log / verdict step reached / timeout.
set -u
LOG="${1:?log file}"
VERDICT_STEP="${2:-14400}"
MAX_S="${3:-2940}"
# repo-root relative to THIS script (works on the local 4090 and the remote box)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

END=$(( $(date +%s) + MAX_S ))
REASON="timeout"
while [ $(date +%s) -lt $END ]; do
    live=0
    for pid in $(pgrep -f "v13_sempty\.train" 2>/dev/null); do
        exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || echo "")
        case "$exe" in *python*) live=1; break;; esac
    done
    if [ "$live" -eq 0 ]; then REASON="TRAIN-PROCESS-GONE"; break; fi

    grep -qE "OutOfMemoryError|CUDA out of memory|Traceback \(most recent|nan" "$LOG" 2>/dev/null && { REASON="TRAIN-ERROR-IN-LOG"; break; }

    STEP=$(grep -hoE "^step [0-9]+" "$LOG" 2>/dev/null | awk '{print $2}' | tail -1)
    [ -n "$STEP" ] && [ "$STEP" -ge "$VERDICT_STEP" ] && { REASON="VERDICT-POINT-REACHED"; break; }

    sleep 60
done
echo "WATCHDOG EXIT: $REASON  (epoch=$(date +%s))"
echo "--- val lines ---"
grep -hE "\[val @" "$LOG" 2>/dev/null | tail -8
echo "--- latest step lines ---"
grep -hE "^step " "$LOG" 2>/dev/null | tail -4
tmux ls 2>/dev/null | grep -E "sempty" || echo "no sempty tmux sessions"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
