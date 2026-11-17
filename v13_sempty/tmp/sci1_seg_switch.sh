#!/bin/bash
# SCI-1 segment switch on the 6000 (generic; used for 3d -> 3r fallback or 3d -> 3e): waits for the NEXT latest.pt save of the running
# segment, stops its tmux, renames that save to <dir>/<label>_end.pt, starts the new segment from it with <env> (sourced after
# sci1_env.sh) from THIS worktree's code. Launch ONLY on Lead's GO.
#   usage: sci1_seg_switch.sh <run_dir under checkpoints_v13_sempty> <tmux of running segment> <env file> <label>
#   (6000, new worktree @ <hash>) tmux new-session -d -s sci1s3r "bash v13_sempty/tmp/sci1_seg_switch.sh sci1_long3d_cf9dee1 sci1s3d \
#       v13_sempty/tmp/sci1_long3r_env.sh seg3d 2>&1 | tee -a logs/v13_sempty_sci1_seg_switch_<hash>_<stamp>.log"
cd "$(dirname "$0")/../.." || exit 1
RUN=${1:?run dir}; TM=${2:?tmux}; ENV=${3:?env}; LABEL=${4:?label}
D=checkpoints_v13_sempty/$RUN; L=$D/latest.pt; START=$D/${LABEL}_end.pt
[ -f "$ENV" ] || { echo "no $ENV"; exit 1; }
[ -e "$START" ] && { echo "$START exists, STOP"; exit 1; }
[ "$(sha256sum .cache/decontam/v2.npy | cut -c1-8)" = a4b66d1b ] || { echo "decontam v2 sha mismatch, STOP"; exit 1; }
t0=$(stat -c %Y $L); echo "=== switch: waiting for the next save of $L (now $(date -u +%FT%TZ))"
until [ "$(stat -c %Y $L 2>/dev/null || echo $t0)" != "$t0" ]; do sleep 10; done
a=0; b=1; until [ "$a" = "$b" ]; do a=$(stat -c %s $L); sleep 20; b=$(stat -c %s $L); done      # save finished
tmux kill-session -t "=$TM" 2>/dev/null; echo "=== switch: $TM stopped $(date -u +%FT%TZ)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done
mv $L $START
.venv/bin/python -c "
import torch; c = torch.load('$START', map_location='cpu', weights_only=False, mmap=True); dc = (c['data_cursor'] or {}).get('doc_counters', {})
print('start step', c['global_step'], '| far / chat cursors', {k: v for k, v in dc.items() if k.split('@')[0] in ('proto_far', 'proto_far2', 'b2s_far', 'smoltalk2_mid', 'chatmt2', 'chatsr', 'tuluif')})" || exit 1
source v13_sempty/tmp/sci1_env.sh && source $ENV && export CONT_FROM=$START && bash v13_sempty/tmp/sci1_cont_chain.sh
