#!/bin/bash
# Free-space gate (Lead 2026-11-17, after the d_drive filled to 100 % and the wio gdn decay crashed at its save):
#   scripts/require_free_gb.sh <path> <GB> [incoming_bytes]
# exits 0 only if the filesystem holding <path> keeps >= <GB> GB free AFTER an incoming write of [incoming_bytes] (default 0).
# Use before every training launch (saves) and every copy onto d_drive / the 6000; refuse on non-zero.
p=${1:?path}; need=${2:-10}; inc=${3:-0}
avail=$(df --output=avail -B1 "$(readlink -f "$p")" 2>/dev/null | tail -1 | tr -dc 0-9)
[ -n "$avail" ] || { echo "require_free_gb: cannot stat $p" >&2; exit 2; }
left=$(( (avail - inc) / 1073741824 ))
[ $left -ge $need ] && exit 0
echo "require_free_gb: REFUSE: $p would have ${left} GB free (< ${need} GB) after ${inc} bytes" >&2; exit 1
