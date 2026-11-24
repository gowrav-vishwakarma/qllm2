#!/bin/bash
# sft1 one-command launcher (6000, from a worktree at the sft_mix commit). Needs the base on this box:
#   checkpoints_v13_sempty/sci1_long3r_91a87ed/final_step_749500.pt  (sha256 b403ad63... as backed up on the 4090 d_drive)
#   tmux new-session -d -s sci1sft1 "bash v13_sempty/tmp/sft1_launch.sh 2>&1 | tee -a logs/v13_sempty_sci1_sft1_launch_<hash>_<stamp>.log"
cd "$(dirname "$0")/../.." || exit 1
B=checkpoints_v13_sempty/sci1_long3r_91a87ed/final_step_749500.pt
[ -s $B ] || { echo "no $B on this box: copy it first (sha b403ad63...)"; exit 1; }
echo "sft1 base sha256 $(sha256sum $B | cut -c1-16)"
source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_long3r_env.sh && source v13_sempty/tmp/sft1_env.sh \
  && export CONT_FROM=$B && bash v13_sempty/tmp/sci1_cont_chain.sh
