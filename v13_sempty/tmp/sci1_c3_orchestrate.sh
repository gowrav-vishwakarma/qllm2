#!/bin/bash
# s8reg C3 on the 6000: pause the long run right after its next latest.pt save, sha-check M9, run C3 (sci1_c3_env.sh), then relaunch
# the long run at its own pinned code (../qllm_sci1 @ 3c2bf1f); it resumes from its own latest.pt (no reset). The local runner
# v13_sempty/tmp/s8reg_c3.sh copies each C3 save, sha-verifies it and deletes it here, so the 25 GB floor holds for the relaunch.
cd "$(dirname "$0")/../.." || exit 1
S=/home/ubuntu/Development/qllm_sci1; L=$S/checkpoints_v13_sempty/sci1_long_3c2bf1f
gpu_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
[ "$(sha256sum checkpoints_v13_sempty/sci1_cont3_449e320/step_137331.pt | cut -c1-8)" = 218922f9 ] || { echo "M9 sha mismatch, STOP"; exit 1; }
t0=$(stat -c %Y $L/latest.pt); echo "=== c3 orch: waiting for the long run's next latest.pt save $(date -u +%FT%TZ)"
until [ "$(stat -c %Y $L/latest.pt)" -gt "$t0" ] && [ ! -e $L/latest.pt.tmp ]; do sleep 20; done
sleep 20; tmux kill-session -t "=sci1"; echo "=== c3 orch: long run paused $(date -u +%FT%TZ)"
until [ "$(gpu_mb)" -lt 2000 ]; do sleep 10; done
( source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_cont3_env.sh && source v13_sempty/tmp/sci1_c3_env.sh && \
  bash v13_sempty/tmp/sci1_decay_from.sh ); echo "=== c3 orch: C3 exit=$? $(date -u +%FT%TZ)"
until [ "$(gpu_mb)" -lt 2000 ]; do sleep 10; done
until [ "$(df -BG / | awk 'NR==2{gsub("G","",$4); print $4}')" -ge 26 ]; do echo "=== c3 orch: disk < 26 GB, waiting for the local copier $(date -u +%FT%TZ)"; sleep 300; done
tmux new-session -d -s sci1 "cd $S && source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_cont3_env.sh && source v13_sempty/tmp/sci1_long_env.sh && bash v13_sempty/tmp/sci1_cont_chain.sh 2>&1 | tee -a logs/v13_sempty_sci1_long_chain_3c2bf1f_\$(date -u +%Y%m%d_%H%M).log"
echo "=== c3 orch: long run relaunched (resumes from its latest.pt) $(date -u +%FT%TZ)"
