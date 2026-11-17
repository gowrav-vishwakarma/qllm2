#!/bin/bash
# Lead's 43B decision rule (EXPERIMENTS_SEMPY "DECISION RULE at 43B", recorded 12:05 IST 2026-11-17 before the number): read fu 8k from
# Library's "3d 656137 told_margin8k n ... fu x" line; fu >= 0.30 -> launch the 3d -> 3e switch on the 6000 (pre-given GO); else no switch.
W=/home/gowrav/Development/qllm_v14norm/logs/v14_fartold_watch.log
until grep -qE "3d 656137 told_margin8k n" $W; do sleep 30; done
LINE=$(grep -E "3d 656137 told_margin8k n" $W | tail -1); echo "43B line: $LINE"
FU=$(echo "$LINE" | sed -E 's/^.*told_margin8k n [0-9]+ [^|]* fu ([0-9.]+).*$/\1/')
echo "fu 8k = $FU ($(date))"
python3 -c "import sys; sys.exit(0 if float('$FU') >= 0.30 else 1)" || { echo "RULE (B): fu < 0.30, no switch"; exit 0; }
echo "RULE (A): fu >= 0.30 ($(date))"
# HOLD (Library / Lead 12:18 IST: held-out scope direction may patch proto_far2): no launch while ~/.qllm_3e_hold exists;
# the worktree must be at the commit named in ~/.qllm_3e_hash (re-synced on the 6000 before launch)
while [ -f $HOME/.qllm_3e_hold ]; do sleep 30; done
WANT=$(cat $HOME/.qllm_3e_hash 2>/dev/null)
if [ -n "$WANT" ]; then ssh -o ConnectTimeout=20 ubuntu@34.131.203.207 "cd ~/Development/qllm-private && git fetch -q origin && cd ../qllm_sci1e && git checkout -q --detach $WANT && git log --oneline -1" || { echo "worktree sync to $WANT failed, STOP"; exit 1; }; fi
echo "launching the 3e switch on the 6000 ($(date))"
STAMP=$(date -u +%Y%m%d_%H%M)
ssh -o ConnectTimeout=20 ubuntu@34.131.203.207 "cd ~/Development/qllm_sci1e && H=\$(git rev-parse --short=8 HEAD) && tmux new-session -d -s sci1s3e \"bash v13_sempty/tmp/sci1_seg_switch.sh sci1_long3d_cf9dee1 sci1s3d v13_sempty/tmp/sci1_long3e_env.sh seg3d 2>&1 | tee -a logs/v13_sempty_sci1_seg_switch_\${H}_$STAMP.log\" && sleep 5 && tmux ls | grep sci1s3e && echo launched \$H"
