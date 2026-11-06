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
  sleep 180
  # free the 4090 first (2026-11-06: the M10 CHECK OOM'd beside cx_m2): pause cx_m2, wait for >= 20 GB free, resume it after
  CX=/home/gowrav/Development/qllm_cx/checkpoints_v13_sempty/cx_m2_main_c17775a; cxp=0
  if tmux has-session -t "=mindcxm2" 2>/dev/null; then cxp=1; touch $CX/.PAUSED; tmux kill-session -t "=mindcxm2"; sleep 5
    P=$(pgrep -f "v13_sempty.train .*--checkpoint_dir checkpoints_v13_sempty/cx_m2_main_c17775a( |$)" | head -1)
    [ -n "$P" ] && { kill -TERM $P; sleep 30; kill -KILL $P 2>/dev/null; }; fi
  until [ "$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F", " '{print $1-$2}')" -ge 20000 ]; do sleep 30; done
  RUN=$RUN bash v13_sempty/tmp/sci1_check_local.sh $m; echo "$RUN $m exit $? $(TZ=Asia/Kolkata date)"
  # then Library's FAR-TOLD for this milestone runs on the free card (Library 2026-11-07); cx_m2 resumes after its DONE line (<= 60 min)
  K=$(( ${m#step_} / 15259 )); FT=/home/gowrav/Development/qllm_v14norm/logs/v14_fartold_watch.log; t=0
  until grep -q "FAR-TOLD-m${K}far-DONE" $FT 2>/dev/null || [ $t -ge 120 ]; do sleep 30; t=$((t + 1)); done
  grep -q "FAR-TOLD-m${K}far-DONE" $FT 2>/dev/null && echo "FAR-TOLD m$K done" || echo "FAR-TOLD m$K not done after 60 min, resuming cx_m2 anyway"
  if [ $cxp = 1 ]; then rm -f $CX/.PAUSED; tmux new-session -d -s mindcxm2 "cd /home/gowrav/Development/qllm_cx && SCREEN_HASH=c17775a NEED_CX=16000 bash v16/complex/cx_chain_live.sh 2>&1 | tee -a logs/v16_cx_chain_c17775a.log"; echo "cx_m2 resumed"; fi
done
