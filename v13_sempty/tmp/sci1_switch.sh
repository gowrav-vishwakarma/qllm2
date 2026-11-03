#!/bin/bash
# SCI-1 segment switch, run ON THE 6000 in tmux (survives the local box): wait until the old trainer has written milestone $1 and logged
# past it, stop the old chain (tmux session sci1), free disk (old latest.pt is superseded by the milestone; old milestones that have a
# verified local copy on the 4090 are listed in DROP_OK), check out the new code $2 in ~/Development/qllm_sci1, launch the continuation
# chain in tmux sci1.   bash v13_sempty/tmp/sci1_switch.sh step_061036 <newhash> [old_run_dir]
set -u
M=$1; NEWH=$2; OLD=${3:-checkpoints_v13_sempty/sci1_main_59c3cf5}; MS=$((10#${M#step_}))
cd ~/Development/qllm_sci1 || exit 1
L=$(ls -t logs/v13_sempty_sci1_main_59c3cf5_*.log | head -1)
echo "switch: waiting for $OLD/$M.pt and a log step > $MS in $L ($(date -u +%FT%TZ))"
until [ -s $OLD/$M.pt ] && [ "$(grep -E '^step [0-9]+ ' $L | tail -1 | awk '{print $2}')" -gt $MS ] 2>/dev/null; do sleep 30; done
s1=$(stat -c %s $OLD/$M.pt); sleep 20; s2=$(stat -c %s $OLD/$M.pt); [ "$s1" = "$s2" ] || { echo "switch: $M still growing"; exit 1; }
.venv/bin/python - <<PY || { echo "switch: $M does not load, ABORT (old run left running)"; exit 1; }
import torch; ck = torch.load('$OLD/$M.pt', map_location='cpu', weights_only=False, mmap=True)
assert ck['global_step'] == $MS, ck['global_step']; print('switch: milestone ok', ck['global_step'], ck.get('global_tokens'))
PY
echo "switch: stopping old chain $(date -u +%FT%TZ)"; tmux kill-session -t sci1
for i in $(seq 60); do pgrep -f "v13_sempty.train .*sci1_main" >/dev/null || break; sleep 5; done
pgrep -f "v13_sempty.train .*sci1_main" && { pkill -TERM -f "v13_sempty.train .*sci1_main"; sleep 30; }
rm -f $OLD/latest.pt; for f in ${DROP_OK:-}; do rm -f $OLD/$f; done
git fetch -q origin && git checkout -q $NEWH || { echo "switch: checkout $NEWH failed"; exit 1; }
H=$(git rev-parse --short HEAD); T=$(date -u +%Y%m%d_%H%M); df -BG / | tail -1
tmux new-session -d -s sci1 "source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_cont1_env.sh && bash v13_sempty/tmp/sci1_cont_chain.sh 2>&1 | tee -a logs/v13_sempty_sci1_cont1_chain_${H}_${T}.log"
sleep 5; tmux ls; echo "switch: launched cont1 at $H $(date -u +%FT%TZ)"
for i in $(seq 90); do NL=$(ls -t logs/v13_sempty_sci1_cont1_${H}_*.log 2>/dev/null | head -1)
  [ -n "$NL" ] && grep -qE '^step [0-9]+ ' $NL && { echo "switch: cont1 training: $(grep -E '^step ' $NL | tail -1 | cut -c1-90)"; exit 0; }; sleep 20; done
echo "switch: FAIL - no training step in the cont1 log after 30 min $(date -u +%FT%TZ)"; exit 1
