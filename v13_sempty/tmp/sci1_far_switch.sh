#!/bin/bash
# SCI-1 far-block switch on the 6000 (EXPERIMENTS_SEMPY "far block"): wait for the long run's M11 (sci1_long_3c2bf1f/step_167849.pt),
# stop the long run, read the checkpoint's cumulative token counters (= RAMP_FROM_TOKENS), and start segment 2 (sci1_long2_env.sh:
# cont3 + the far block ramped in over 2B, WARMUP 0) from THIS worktree's code.
#   (6000, worktree ../qllm_sci1b @ <hash>) tmux new-session -d -s sci1far "bash v13_sempty/tmp/sci1_far_switch.sh 2>&1 | tee -a logs/..."
cd "$(dirname "$0")/../.." || exit 1
L=/home/ubuntu/Development/qllm_sci1/checkpoints_v13_sempty/sci1_long_3c2bf1f
[ -s .cache/v7_tokens/b2far_filler_dclm_train_e3_p5_20k_v1.npy ] || { echo "no b2s_far filler pool, STOP"; exit 1; }
echo "=== far switch: waiting for $L/step_167849.pt $(date -u +%FT%TZ)"
until [ -s $L/step_167849.pt ]; do sleep 60; done
sleep 90; tmux kill-session -t "=sci1"; echo "=== far switch: long run (segment 1) stopped at M11 $(date -u +%FT%TZ)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 10; done
export RAMP_FROM_TOKENS=$(.venv/bin/python -c "
import torch; c = torch.load('$L/step_167849.pt', map_location='cpu', weights_only=False, mmap=True)
print(sum(v for k, v in (c['data_cursor'] or {}).get('token_counters', {}).items() if '#' not in k and '@' not in k))")
echo "=== far switch: RAMP_FROM_TOKENS=$RAMP_FROM_TOKENS $(date -u +%FT%TZ)"
source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_long2_env.sh && bash v13_sempty/tmp/sci1_cont_chain.sh
