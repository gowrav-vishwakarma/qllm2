#!/bin/bash
# SCI-1 milestone CHECK on the 4090 (the 6000's trainer holds ~92 GB, so the CHECK cannot run beside it): copy the next
# unchecked milestone from the 6000, run v13_sempty/tmp/sci1_check.sh at the run's code (worktree ../qllm_sci1check @ 59c3cf5),
# then exit (Mind commits the outputs). One milestone per call.
#   bash v13_sempty/tmp/sci1_check_local.sh [step_XXXXXX]
set -e
RUN=${RUN:-sci1_main_59c3cf5}   # the segment dir (cont1: RUN=sci1_cont1_<hash>)
R=ubuntu@34.131.203.207; RD=Development/qllm_sci1/checkpoints_v13_sempty/$RUN
LD=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/$RUN; mkdir -p $LD
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
cp /home/gowrav/Development/qllm2/v13_sempty/tmp/sci1_check.sh $W/v13_sempty/tmp/sci1_check.sh   # the master copy (MEM_FRAC for the 4090)
cp /home/gowrav/Development/qllm2/scripts/sci_gen.py $W/scripts/sci_gen.py   # no reasoning switch (thinking off, 2026-11-03)
cd $W
# RAM guard (the local box hung 2026-11-02 23:35; cause unproven): kill the CHECK if MemAvailable < 12 GB
( MEM_FRAC=0.5 setsid bash v13_sempty/tmp/sci1_check.sh $LD/$m.pt sci1_${m}_${RUN##*_} >> logs/v13_sempty_sci1_checks_59c3cf5_local.log 2>&1 ) &
CP=$!
while kill -0 $CP 2>/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$a" -lt 12 ]; then echo "RAM GUARD: MemAvailable ${a} GB < 12, killing the CHECK $(date -u +%FT%TZ)" | tee -a logs/v13_sempty_sci1_checks_59c3cf5_local.log
    pkill -TERM -f "sci1_${m}_${RUN##*_}"; exit 1; fi
  sleep 10
done
touch $LD/.checked_$m; echo "=== local CHECK $m DONE $(date -u +%FT%TZ)"
