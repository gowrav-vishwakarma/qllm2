#!/bin/bash
# s8reg C2 on the 6000 (Lead GO 2026-11-06): wait for the long run's M10 (step_152590.pt, written atomically), pause the long run,
# run the C2 decay (0.25B, rule block x2, saves every ~0.05B) from this worktree, then resume the long run EXACTLY at M10
# (latest.pt <- the full step_152590.pt) in tmux sci1 at its own pinned code (../qllm_sci1 @ 3c2bf1f).
#   (6000, worktree ../qllm_c2) tmux new-session -d -s sci1c2 "bash v13_sempty/tmp/sci1_c2_orchestrate.sh 2>&1 | tee -a logs/v13_sempty_sci1_c2_orch_<hash>.log"
cd "$(dirname "$0")/../.." || exit 1
S=/home/ubuntu/Development/qllm_sci1; L=$S/checkpoints_v13_sempty/sci1_long_3c2bf1f
gpu_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
echo "=== c2 orch: waiting for $L/step_152590.pt $(date -u +%FT%TZ)"
until [ -s $L/step_152590.pt ]; do sleep 60; done
sleep 90; echo "=== c2 orch: M10 saved; pausing the long run $(date -u +%FT%TZ)"
tmux kill-session -t "=sci1"; sleep 30
until [ "$(gpu_mb)" -lt 2000 ]; do sleep 10; done
( source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_cont3_env.sh && source v13_sempty/tmp/sci1_c2_env.sh && \
  bash v13_sempty/tmp/sci1_decay_from.sh ); echo "=== c2 orch: C2 exit=$? $(date -u +%FT%TZ)"
until [ "$(gpu_mb)" -lt 2000 ]; do sleep 10; done
cp $L/step_152590.pt $L/latest.pt.c2tmp && mv $L/latest.pt.c2tmp $L/latest.pt
tmux new-session -d -s sci1 "cd $S && source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_cont3_env.sh && source v13_sempty/tmp/sci1_long_env.sh && bash v13_sempty/tmp/sci1_cont_chain.sh 2>&1 | tee -a logs/v13_sempty_sci1_long_chain_3c2bf1f_\$(date -u +%Y%m%d_%H%M).log"
echo "=== c2 orch: long run resumed at M10 $(date -u +%FT%TZ)"
