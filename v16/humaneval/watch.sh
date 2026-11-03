#!/bin/bash
# Wait for Mind's CHECK marker of each milestone, then build that checkpoint's human-eval page and serve it.
#   tmux new-session -d -s he_watch "bash v16/humaneval/watch.sh sci1_cont1_59d098d 076295 091554"
# Gated on .checked_step_<N> so it never overlaps the CHECK on the 4090; reads the checkpoint in place (never deletes it).
H=$(cd "$(dirname "$0")" && pwd); RUN=$1; shift
LD=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/$RUN
for S in "$@"; do
  echo "waiting for $LD/.checked_step_$S $(date -u +%FT%TZ)"
  until [ -e $LD/.checked_step_$S ]; do sleep 60; done
  bash $H/run.sh $LD/step_$S.pt || { echo "run failed for $S"; continue; }
  out=$(bash $H/run.sh serve $S); echo "$out"; U=${out##*-> }
  [ -n "$U" ] && [ -n "$NTFY" ] && curl -s -o /dev/null -d "UPDATE ($(TZ=Asia/Kolkata date +%H:%M) IST): human-eval page for SCI-1 step $S is ready: $U" "$NTFY"
done
