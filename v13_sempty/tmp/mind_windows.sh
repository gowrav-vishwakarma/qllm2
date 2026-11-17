#!/bin/bash
# (generic: TARGETS env) 4090 windows for Library's 3f scope probes (Library 2026-11-17): for each 3f save, once it is on d_drive, take the card from the wio
# chain (yield flag), open ~/.qllm_lib_window, close both on Library's "window done 3f <T>" line. Hard stop 10:30 IST 11-18 (Library).
D=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/sci1_long3f_91a87ed
W=/home/gowrav/Development/qllm_v14norm/logs/v14_fartold_watch.log; WL=/home/gowrav/Development/qllm2/logs/v13_sempty_sci1_waiter_local.log
STOP=$(date -d "2026-11-18 10:30" +%s)
waiter_idle() { local l; l=$(tail -1 $WL); [[ "$l" == waiting\ for* || "$l" == *" exit "* || "$l" == FAR-TOLD* ]]; }   # no CHECK pending / running
for T in ${TARGETS:?}; do
  if [ $T = 686655 ]; then until [ -f $D/.checked_step_686655 ] && [ -s $D/step_686655.pt ]; do sleep 60; done
  else until [ -f $D/.snap_$T.done ] && [ -e $D/snap_$T.pt ]; do [ -f $D/.snap_$T.skipped ] && continue 2; sleep 60; done; fi
  [ "$(date +%s)" -ge $STOP ] && { echo "past 10:30, no more windows"; exit 0; }
  grep -q "window done 3f $T" $W && { echo "$T already scored"; continue; }
  touch ~/.qllm_check_wants_gpu; echo "=== $T: flag up $(date +%T)"
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 4000 ]; do sleep 15; done   # wio trainer gone
  touch ~/.qllm_lib_window; echo "=== $T: window open $(date +%T)"
  until grep -q "window done 3f $T" $W || [ "$(date +%s)" -ge $STOP ]; do [ -f ~/.qllm_check_wants_gpu ] || touch ~/.qllm_check_wants_gpu; sleep 15; done
  rm -f ~/.qllm_lib_window
  if waiter_idle && ! pgrep -f '^bash v13_sempty/tmp/sci1_check_local.sh' >/dev/null; then rm -f ~/.qllm_check_wants_gpu; echo "=== $T: window closed, card back to wio $(date +%T)"
  else echo "=== $T: window closed; a milestone CHECK owns the flag $(date +%T)"; fi
done; echo "=== 3f windows done $(date)"
