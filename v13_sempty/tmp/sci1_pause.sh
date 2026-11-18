#!/bin/bash
# SCI-1 pause on the 6000 (Lead 11-18: 6000 test window): wait for the NEXT latest.pt save of the running segment, stop its tmux,
# rename that save to <dir>/<label>.pt (e.g. seg3r_pause.pt), print its step. Resume later with sci1_cont_chain.sh CONT_FROM=<that file>
# and the segment's env (all cursors kept).   usage: sci1_pause.sh <run_dir> <tmux> <label>   (run from the segment's worktree)
cd "$(dirname "$0")/../.." || exit 1
RUN=${1:?run dir}; TM=${2:?tmux}; LABEL=${3:?label}
D=checkpoints_v13_sempty/$RUN; L=$D/latest.pt; P=$D/$LABEL.pt
[ -e "$P" ] && { echo "$P exists, STOP"; exit 1; }
t0=$(stat -c %Y $L); echo "=== pause: waiting for the next save of $L (now $(date -u +%FT%TZ))"
until [ "$(stat -c %Y $L 2>/dev/null || echo $t0)" != "$t0" ]; do sleep 10; done
a=0; b=1; until [ "$a" = "$b" ]; do a=$(stat -c %s $L); sleep 20; b=$(stat -c %s $L); done
tmux kill-session -t "=$TM" 2>/dev/null; echo "=== pause: $TM stopped $(date -u +%FT%TZ)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done
mv $L $P
.venv/bin/python -c "
import torch; c = torch.load('$P', map_location='cpu', weights_only=False, mmap=True)
print('=== PAUSED at step', c['global_step'], 'tokens', c['global_tokens'], '->', '$P')" || exit 1
