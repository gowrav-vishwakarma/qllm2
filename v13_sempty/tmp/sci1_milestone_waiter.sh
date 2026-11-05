#!/bin/bash
# SCI-1 local milestone waiter (survives in git; the scratchpad copies are lost on a reboot): for each milestone step, wait until the
# 6000 has <run>/step_X.pt, then run the 4090 CHECK (v13_sempty/tmp/sci1_check_local.sh copies it to d_drive on its own hard link and
# touches .checked_step_X, which Lead's humaneval and Library's A26 gate on). One milestone at a time.
#   tmux new-session -d -s mindm9 "bash v13_sempty/tmp/sci1_milestone_waiter.sh sci1_cont3_449e320 step_137331 2>&1 | tee -a logs/v13_sempty_sci1_waiter_local.log"
RUN=$1; shift
cd "$(dirname "$0")/../.." || exit 1
for m in "$@"; do
  L=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/$RUN
  [ -f $L/.checked_$m ] && { echo "$RUN $m already checked"; continue; }
  echo "waiting for $RUN/$m.pt on the 6000 $(TZ=Asia/Kolkata date)"
  until ssh -o ConnectTimeout=20 ubuntu@34.131.203.207 "ls ~/Development/qllm_sci1/checkpoints_v13_sempty/$RUN/$m.pt" >/dev/null 2>&1; do sleep 300; done
  sleep 180; RUN=$RUN bash v13_sempty/tmp/sci1_check_local.sh $m; echo "$RUN $m exit $? $(TZ=Asia/Kolkata date)"
done
