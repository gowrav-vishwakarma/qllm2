#!/usr/bin/env bash
# DELETION SENTINEL — watches git status for deleted tracked files (2nd incident
# 2026-08-24 12:35: v13 core + v7/train.py deleted twice by an unknown external
# process). On first detection: records evidence (timestamp + newest processes),
# auto-restores from HEAD, exits. Re-arm every wake while the threat persists.
# Usage: bash v13/tmp/deletion_sentinel.sh [max_seconds=3300]
set -u
cd /home/gowrav/Development/qllm2
MAX_S="${1:-3300}"
END=$(( $(date +%s) + MAX_S ))
while [ $(date +%s) -lt $END ]; do
  DEL=$(git status --porcelain 2>/dev/null | awk '$1 ~ /^( D|D )/ {print $2}' | grep -E '^(v11|v13|v7|scripts)/' || true)
  if [ -n "$DEL" ]; then
    echo "SENTINEL: DELETIONS DETECTED at $(date '+%F %T')"
    echo "--- deleted ---"
    echo "$DEL"
    echo "--- newest processes (start_time) ---"
    ps -eo pid,ppid,lstart,args --sort=start_time 2>/dev/null | tail -15
    echo "--- cursor/omp/agent procs ---"
    ps aux | grep -iE "omp|cursor-agent|claude|codex|gemini" | grep -v grep | awk '{print $2, $9, $10, $11, $12, $13}'
    echo "--- restoring from HEAD ---"
    git checkout HEAD -- v13/ v7/ v11/ scripts/ 2>&1
    echo "SENTINEL: restored, exiting"
    exit 0
  fi
  sleep 20
done
echo "SENTINEL: clean for full window ($(date '+%F %T'))"
