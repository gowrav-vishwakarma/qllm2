#!/bin/bash
# SCI-1 segment-3 switch on the 6000 (EXPERIMENTS_SEMPY "segment 3"): FROM=28b waits for segment 2's 28B milestone (step_427252.pt);
# FROM=344k starts from segment 2's best_model (step 344,000, before fineweb ran dry; Lead's rollback rule, decided by told 8k).
# Stops segment 2, builds + sha-checks decontam index v2, reads RAMP_FROM_TOKENS from the start checkpoint, starts segment 3
# (sci1_long3_env.sh) from THIS worktree's code.
#   (6000, worktree ../qllm_sci1c @ <hash>) FROM=28b|344k tmux new-session -d -s sci1s3 "bash v13_sempty/tmp/sci1_seg3_switch.sh 2>&1 | tee -a logs/..."
cd "$(dirname "$0")/../.." || exit 1
: "${FROM:?FROM=28b or 344k}"
D=checkpoints_v13_sempty/sci1_long2_4a97ca7
if [ ! -s .cache/decontam/v2.npy ]; then .venv/bin/python v14/decontam_v2.py || exit 1; fi
[ "$(sha256sum .cache/decontam/v2.npy | cut -c1-8)" = a4b66d1b ] || { echo "decontam v2 sha mismatch, STOP"; exit 1; }
if [ "$FROM" = 344k ]; then
  [ -s $D/step_344000_best.pt ] || cp $D/best_model.pt $D/step_344000_best.pt
  START=$D/step_344000_best.pt
else
  START=$D/step_427252.pt; echo "=== seg3 switch: waiting for $START $(date -u +%FT%TZ)"
  until [ -s $START ]; do sleep 60; done; sleep 90
fi
.venv/bin/python -c "import torch; c = torch.load('$START', map_location='cpu', weights_only=False, mmap=True); print('start step', c['global_step'])" || exit 1
tmux kill-session -t "=sci1far" 2>/dev/null; tmux kill-session -t "=sci1" 2>/dev/null; echo "=== seg3 switch: segment 2 stopped; start = $START $(date -u +%FT%TZ)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done
export RAMP_FROM_TOKENS=$(.venv/bin/python -c "
import torch; c = torch.load('$START', map_location='cpu', weights_only=False, mmap=True)
print(sum(v for k, v in (c['data_cursor'] or {}).get('token_counters', {}).items() if '#' not in k and '@' not in k))")
echo "=== seg3 switch: RAMP_FROM_TOKENS=$RAMP_FROM_TOKENS $(date -u +%FT%TZ)"
source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_long3_env.sh && export CONT_FROM=$START && bash v13_sempty/tmp/sci1_cont_chain.sh
