#!/bin/bash
# 6000: seed-45 trio (pam, nochrono, split; wio far mix; Lead 11-19) via twins_chain.sh BOX=6000, one instance per arm, each started
# only when fewer than 3 trainers run (distinct --checkpoint_dir, so data workers are not counted) and >= 21.8 GB VRAM is free.
cd "$(dirname "$0")/../.." || exit 1
H=$(git rev-parse --short HEAD)
ntrain() { ps -eo args | grep "[v]13_sempty.train" | grep -o "checkpoint_dir [^ ]*" | sort -u | wc -l; }
freemb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
for spec in "pam_s45|CHRONO=1 SEED=45 \$PAMSPEC" "nochrono_s45|CHRONO=0 SEED=45 \$PAMSPEC" "split_s45|CHRONO=1 CHRONO_HEADS=3 SEED=45 \$PAMSPEC"; do
  a=${spec%%|*}
  until [ "$(ntrain)" -lt 3 ] && [ "$(freemb)" -ge 21800 ]; do sleep 120; done
  echo "=== launch $a $(date -u +%FT%TZ) (trainers $(ntrain), free $(freemb) MiB)"
  tmux new-session -d -s tw_$a "cd ~/Development/qllm_twins && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ALLOW_CODE_DRIFT=1 BOX=6000 SCREEN_HASH=$H CHAIN_ARMS='$spec' bash v13_sempty/tmp/twins_chain.sh 2>&1 | tee -a logs/v13_sempty_twins_${a}_chain_${H}_$(date -u +%Y%m%d_%H%M).log"
  until [ "$(ntrain)" -ge 1 ] && ps -eo args | grep -q "[v]13_sempty.train .*nt_${a}_main"; do sleep 60; done   # its trainer is up
done; echo "=== seed45 launcher done $(date -u +%FT%TZ)"
