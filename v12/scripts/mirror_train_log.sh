#!/usr/bin/env bash
# Periodically snapshot a live v12.train TeeLogger fd into a durable file.
#
# Why: if the on-disk log path is deleted/recreated while train still holds the
# old inode, progress only lives under /proc/<pid>/fd/<n>. When that process
# exits the inode is freed — this script keeps a last-good copy on disk.
#
# On train exit / missing fd: leave the output file untouched and exit 0
# (no error text, no truncate).
#
# Usage:
#   v12/scripts/mirror_train_log.sh [OUT_LOG]
# Env:
#   PID          — pin a train PID (default: auto-detect v12.train)
#   INTERVAL     — seconds between snapshots (default 10)
#   MATCH        — substring to find in /proc/PID/fd targets (default pretrain_fact.log)
#   DATASET      — optional --dataset filter for auto-detect (e.g. fact, pretrain_mix)

set -u
cd "$(cd "$(dirname "$0")/../.." && pwd)"

OUT="${1:-logs/v12_smoke_fact/04_fact_additive_live.log}"
INTERVAL="${INTERVAL:-10}"
MATCH="${MATCH:-pretrain_fact.log}"
mkdir -p "$(dirname "$OUT")"

find_train_pid() {
  if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
    echo "$PID"
    return 0
  fi
  # Prefer the parent (non-worker) process: highest CPU / first match of the
  # main module invocation.
  local p ds_pat=""
  if [ -n "${DATASET:-}" ]; then
    ds_pat="&& /--dataset ${DATASET}/"
  fi
  p=$(ps -eo pid=,pcpu=,cmd= \
    | awk "/[.]venv\\/bin\\/python -m v12\\.train / ${ds_pat} && !/cursorsandbox/ {print \$1, \$2}" \
    | sort -k2 -nr \
    | awk 'NR==1 {print $1}')
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"
}

find_log_fd() {
  local pid="$1" fd target
  for fd in /proc/"$pid"/fd/*; do
    [ -e "$fd" ] || [ -L "$fd" ] || continue
    target=$(readlink "$fd" 2>/dev/null) || continue
    case "$target" in
      *"$MATCH"*) basename "$fd"; return 0 ;;
    esac
  done
  return 1
}

snapshot_once() {
  local pid="$1" fd="$2" src tmp
  src="/proc/$pid/fd/$fd"
  tmp="${OUT}.tmp.$$"
  # Never truncate OUT on failure: write tmp first, replace only if non-empty.
  if ! cat "$src" >"$tmp" 2>/dev/null; then
    rm -f "$tmp"
    return 1
  fi
  if [ ! -s "$tmp" ]; then
    rm -f "$tmp"
    return 1
  fi
  # Prefer not to shrink a good snapshot with a partial/racy read.
  if [ -s "$OUT" ]; then
    local old new
    old=$(wc -c <"$OUT")
    new=$(wc -c <"$tmp")
    if [ "$new" -lt "$old" ]; then
      rm -f "$tmp"
      return 0
    fi
  fi
  mv -f "$tmp" "$OUT"
  return 0
}

while true; do
  pid="$(find_train_pid || true)"
  if [ -z "${pid:-}" ]; then
    # Train gone — keep last OUT as-is.
    exit 0
  fi
  if ! fd="$(find_log_fd "$pid")"; then
    # Process still up but log fd gone (finishing / re-opened) — wait a bit,
    # then exit quietly if still missing and process died.
    sleep "$INTERVAL"
    if ! kill -0 "$pid" 2>/dev/null; then
      exit 0
    fi
    continue
  fi
  snapshot_once "$pid" "$fd" || true
  sleep "$INTERVAL"
done
