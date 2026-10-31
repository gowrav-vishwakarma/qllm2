#!/bin/bash
# Wake (resume) a Claude Code session that has exited, in a detached tmux window.
# Learned from Library's restart of Mind (2026-10-31 23:08 IST). Any session may wake any other.
#   scripts/wake_session.sh <Name> <session-id> ["<first prompt>"]
# Known ids (2026-10-31): Mind 99371c3c-8382-42b3-a3d3-6861632abc19 | Library fdbac1f0-c2c3-4b9c-8baf-2585a9e7b8eb
#                         Lead 683d2316-c093-4e15-b19c-044555bd5293 | Evol V1 0001093e-8355-4d42-a4fd-ed097ef58d71
# Check first with ListAgents (or `tmux ls`) that the session is really gone; never start a second copy.
set -u
NAME=$1; SID=$2; PROMPT=${3:-"Resumed by $(whoami)'s Lead/peer session at $(TZ=Asia/Kolkata date +%H:%M) IST (your previous process had exited). Re-read v16/SCRATCHPAD.md (the board) and EUREKA.md, then continue your lane; GPU order goes through Lead."}
B=$(ls -t /home/gowrav/.cursor/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | head -1)
[ -z "$B" ] && B=$(command -v claude)
[ -z "$B" ] && { echo "claude binary not found"; exit 1; }
T=$(echo "$NAME" | tr 'A-Z ' 'a-z_')
if tmux has-session -t "$T" 2>/dev/null; then echo "tmux session $T already exists; not starting a second copy"; exit 2; fi
tmux new-session -d -s "$T" -c /home/gowrav/Development/qllm2 "$B --resume $SID --name '$NAME' '$PROMPT'"
sleep 25; tmux send-keys -t "$T" Enter; sleep 8; tmux send-keys -t "$T" Enter; sleep 8   # pass the theme / notice screens if shown
tmux ls | grep "^$T:"; tmux capture-pane -pt "$T" | grep -v '^\s*$' | tail -6
echo "woken: $NAME in tmux '$T' (attach: tmux attach -t $T). Verify with ListAgents; then message it."
