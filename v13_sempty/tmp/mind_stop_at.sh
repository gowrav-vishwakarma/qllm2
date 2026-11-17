#!/bin/bash
# User needs the 4090 from HH:MM (Lead 2026-11-17: stop at a save by 11:00 IST 2026-11-18): at the given time set the yield flag (the
# wio chain SIGTERMs its trainer; latest.pt is at most 200 steps old) and stop the chain session so nothing relaunches.
AT=${1:?YYYY-mm-dd HH:MM}; until [ "$(date +%s)" -ge "$(date -d "$AT" +%s)" ]; do sleep 30; done
touch ~/.qllm_check_wants_gpu; echo "stop flag set $(date)"; sleep 90; tmux kill-session -t "=mindwio" 2>/dev/null
pkill -TERM -f 'v13_sempty.train .*wio_' 2>/dev/null; echo "wio chain stopped $(date)"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
