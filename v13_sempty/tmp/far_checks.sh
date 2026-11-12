#!/bin/bash
# Far-block read after the 5-day outage (Nov 12): copy + sha-verify the 27B / 26B / M11 / 344k (segment 2 best_model) points, CHECK them
# serially, wait for Library's FAR-TOLD on all four (<= 60 min each), then resume cx_m2. Re-runnable: every step skips finished outputs.
cd /home/gowrav/Development/qllm2
R=ubuntu@34.131.203.207; RD=Development/qllm_sci1/checkpoints_v13_sempty; LD=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty
cp1() { mkdir -p $LD/$1; [ -s $LD/$1/$2 ] && return 0; scp -q $R:$RD/$1/$2 $LD/$1/$2.part && mv $LD/$1/$2.part $LD/$1/$2
  a=$(sha256sum $LD/$1/$2 | cut -c1-64); b=$(ssh -o ConnectTimeout=20 $R "sha256sum ~/$RD/$1/$2 | cut -c1-64")
  [ "$a" = "$b" ] && echo "SHA OK $1/$2 ${a:0:12}" || { echo "SHA MISMATCH $1/$2"; rm -f $LD/$1/$2; }; }
cp1 sci1_long2_4a97ca7 step_411993.pt & cp1 sci1_long2_4a97ca7 step_396734.pt & cp1 sci1_long_3c2bf1f step_167849.pt & cp1 sci1_long2_4a97ca7 best_model.pt & wait
echo "=== copies done $(date -u +%FT%TZ)"
RUN=sci1_long2_4a97ca7 bash v13_sempty/tmp/sci1_check_local.sh step_411993
RUN=sci1_long2_4a97ca7 bash v13_sempty/tmp/sci1_check_local.sh step_396734
RUN=sci1_long_3c2bf1f bash v13_sempty/tmp/sci1_check_local.sh step_167849
RUN=sci1_long2_4a97ca7 bash v13_sempty/tmp/sci1_check_local.sh best_model   # = step 344,000 (22.5B, before fineweb ran dry; Lead's rollback test)
echo "=== CHECKs done $(date -u +%FT%TZ); waiting for FAR-TOLD m27far / m26far / m11far / m22best"
FT=/home/gowrav/Development/qllm_v14norm/logs/v14_fartold_watch.log
for tag in m27far m26far m11far m22best; do t=0; until grep -q "FAR-TOLD-${tag}-DONE" $FT 2>/dev/null || [ $t -ge 120 ]; do sleep 30; t=$((t+1)); done
  grep -q "FAR-TOLD-${tag}-DONE" $FT && echo "FAR-TOLD $tag done" || echo "FAR-TOLD $tag not done after 60 min"; done
rm -f /home/gowrav/Development/qllm_cx/checkpoints_v13_sempty/cx_m2_main_c17775a/.PAUSED
tmux new-session -d -s mindcxm2 "cd /home/gowrav/Development/qllm_cx && SCREEN_HASH=c17775a NEED_CX=16000 bash v16/complex/cx_chain_live.sh 2>&1 | tee -a logs/v16_cx_chain_c17775a.log"
echo "=== cx_m2 resumed $(date -u +%FT%TZ)"
