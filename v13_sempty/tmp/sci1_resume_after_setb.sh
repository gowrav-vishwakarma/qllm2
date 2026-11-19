#!/bin/bash
# 6000 (run from the 3r worktree qllm_sci1e @ 91a87ed): resume SCI-1 segment 3r from seg3r_pause.pt once Set B is DONE and the card is
# empty (the s45 readouts and the lr06 reads finished). Same run dir: latest.pt := hard link to seg3r_pause.pt, so resume=auto restores
# step / cursors / optimizer / schedule; CONT_FROM points at the same file only to satisfy sci1_cont_chain.sh's check.
cd "$(dirname "$0")/../.." || exit 1
D=checkpoints_v13_sempty/sci1_long3r_91a87ed; P=$D/seg3r_pause.pt
TW=/home/ubuntu/Development/qllm_twins/logs
until grep -qs "=== SET B DONE" $(ls -t $TW/v13_sempty_twins_setb_chain_*.log | head -1); do sleep 120; done
echo "=== set B done seen $(date -u +%FT%TZ)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 60; done
[ -s $P ] || { echo "no $P, STOP"; exit 1; }
[ -e $D/latest.pt ] || ln $P $D/latest.pt
echo "=== resuming 3r from $P (step 721,000) $(date -u +%FT%TZ)"
source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_long3r_env.sh && export CONT_FROM=$P && bash v13_sempty/tmp/sci1_cont_chain.sh
