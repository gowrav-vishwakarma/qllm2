#!/bin/bash
# SCI-1 segment-3c switch on the 6000 (EXPERIMENTS_SEMPY "segment 3c"): stops segment 3 (tmux sci1s3), starts segment 3c from segment
# 3's latest save with the chat sources cycling (sci1_long3c_env.sh), from THIS worktree's code.
#   (6000, worktree ../qllm_sci1c @ <hash>) tmux new-session -d -s sci1s3c "bash v13_sempty/tmp/sci1_seg3c_switch.sh 2>&1 | tee -a logs/..."
cd "$(dirname "$0")/../.." || exit 1
D=checkpoints_v13_sempty/sci1_long3_91bdc49
[ "$(sha256sum .cache/decontam/v2.npy | cut -c1-8)" = a4b66d1b ] || { echo "decontam v2 sha mismatch, STOP"; exit 1; }
tmux kill-session -t "=sci1s3" 2>/dev/null; echo "=== seg3c switch: segment 3 stopped $(date -u +%FT%TZ)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done
START=$D/seg3_end.pt; mv $D/latest.pt $START
.venv/bin/python -c "
import torch; c = torch.load('$START', map_location='cpu', weights_only=False, mmap=True); dc = (c['data_cursor'] or {}).get('doc_counters', {})
print('start step', c['global_step'], '| chat cursors', {k: v for k, v in dc.items() if k.split('@')[0] in ('smoltalk2_mid', 'chatmt2', 'chatsr', 'tuluif')})" || exit 1
source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_long3c_env.sh && export CONT_FROM=$START && bash v13_sempty/tmp/sci1_cont_chain.sh
