#!/bin/bash
# SCI-1 milestone CHECK on the 4090 (the 6000's trainer holds ~92 GB, so the CHECK cannot run beside it): copy the next
# unchecked milestone from the 6000, run v13_sempty/tmp/sci1_check.sh at the run's code (worktree ../qllm_sci1check @ 59c3cf5),
# then exit (Mind commits the outputs). One milestone per call.
#   bash v13_sempty/tmp/sci1_check_local.sh [step_XXXXXX]
set -e
R=ubuntu@34.131.203.207; RD=Development/qllm_sci1/checkpoints_v13_sempty/sci1_main_59c3cf5
LD=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/sci1_main_59c3cf5; mkdir -p $LD
W=/home/gowrav/Development/qllm_sci1check; export PATH=/usr/local/cuda/bin:$PATH
m=${1:-}
if [ -z "$m" ]; then
  for f in $(ssh -o ConnectTimeout=20 $R "ls $RD/ | grep -E '^step_[0-9]+\.pt$'"); do
    [ -f $LD/.checked_${f%.pt} ] || { m=${f%.pt}; break; }
  done
fi
[ -n "$m" ] || { echo "no unchecked milestone"; exit 0; }
if [ ! -s $LD/$m.pt ]; then scp -q $R:$RD/$m.pt $LD/$m.pt.part; mv $LD/$m.pt.part $LD/$m.pt; fi
echo "=== local CHECK $m $(date -u +%FT%TZ)"
cd $W && bash v13_sempty/tmp/sci1_check.sh $LD/$m.pt sci1_${m}_59c3cf5 2>&1 | tee -a logs/v13_sempty_sci1_checks_59c3cf5_local.log
touch $LD/.checked_$m; echo "=== local CHECK $m DONE $(date -u +%FT%TZ)"
