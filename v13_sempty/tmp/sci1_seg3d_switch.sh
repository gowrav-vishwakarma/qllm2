#!/bin/bash
# SCI-1 segment-3d switch on the 6000 (EXPERIMENTS_SEMPY "segment 3d"): waits for segment 3c's 41B save (step_625619), stops 3c (tmux
# sci1s3c), starts segment 3d from that save with proto_far 1.6 % (sci1_long3d_env.sh), from THIS worktree's code.
# Launch ONLY after Lead's confirmation on Library's unit4 table (Lead GO condition, 2026-11-16).
#   (6000, worktree ../qllm_sci1c @ <hash>) tmux new-session -d -s sci1s3d "bash v13_sempty/tmp/sci1_seg3d_switch.sh 2>&1 | tee -a logs/v13_sempty_sci1_seg3d_switch_<hash>_<stamp>.log"
cd "$(dirname "$0")/../.." || exit 1
D=checkpoints_v13_sempty/sci1_long3c_3a8d7fb; START=$D/step_625619.pt
[ "$(sha256sum .cache/decontam/v2.npy | cut -c1-8)" = a4b66d1b ] || { echo "decontam v2 sha mismatch, STOP"; exit 1; }
echo "=== seg3d switch: waiting for $START $(date -u +%FT%TZ)"
until [ -s $START ]; do sleep 30; done
a=0; b=1; until [ "$a" = "$b" ]; do a=$(stat -c %s $START); sleep 30; b=$(stat -c %s $START); done   # save finished
tmux kill-session -t "=sci1s3c" 2>/dev/null; echo "=== seg3d switch: segment 3c stopped $(date -u +%FT%TZ)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done
disk_gb() { df -BG / | awk 'NR==2{gsub("G","",$4); print $4}'; }
# 3c's latest.pt is past 41B and superseded by 3d (resume-only); drop it if the 25 GB floor needs the room
[ "$(disk_gb)" -lt 25 ] && [ -f $D/latest.pt ] && { rm $D/latest.pt; echo "removed superseded $D/latest.pt (disk $(disk_gb)G)"; }
.venv/bin/python -c "
import torch; c = torch.load('$START', map_location='cpu', weights_only=False, mmap=True); dc = (c['data_cursor'] or {}).get('doc_counters', {})
print('start step', c['global_step'], '| proto_far / chat cursors', {k: v for k, v in dc.items() if k.split('@')[0] in ('proto_far', 'smoltalk2_mid', 'chatmt2', 'chatsr', 'tuluif')})" || exit 1
source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_long3d_env.sh && export CONT_FROM=$START && bash v13_sempty/tmp/sci1_cont_chain.sh
