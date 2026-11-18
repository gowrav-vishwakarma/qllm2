#!/bin/bash
# 6000 disk: score each Set B step save as soon as it is written (same command + label as setb_chain.sh's scope(), so the chain later
# skips it) and delete the save, instead of holding up to 5 saves per arm until the arm ends. Runs from the qllm_twins worktree.
cd "$(dirname "$0")/../.." || exit 1
H=$(git rev-parse --short HEAD); PY=.venv/bin/python
while true; do
  for k in $(ls checkpoints_v13_sempty/setb_*/step_*.pt 2>/dev/null); do
    [ $(( $(date +%s) - $(stat -c %Y $k) )) -ge 300 ] || continue           # finished writing
    d=$(basename $(dirname $k)); n=${d#setb_}; n=${n%_*}; s=$(basename $k .pt); lab=${n}_${s#step_}
    if ! grep -qs "SCOPE DONE" logs/v14_scope_c_${lab}_*.log; then
      until [ "$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}')" -ge 8000 ]; do sleep 60; done
      o=logs/v14_scope_c_${lab}_${H}_$(date +%Y%m%d_%H%M)
      $PY v14/scope_probe.py --ckpt $k --fullctx --lengths 2048 --docs 100 --device cuda --out $o.jsonl > $o.log 2>&1
      echo "=== reaper scope $lab: $(grep -c Traceback $o.log) tracebacks | $(grep 'L 2048' $o.log | cut -c1-200) $(date -u +%FT%TZ)"
    fi
    grep -qs "SCOPE DONE" logs/v14_scope_c_${lab}_*.log && rm -f $k && echo "=== reaper rm $k"
  done
  [ -f checkpoints_v13_sempty/.setb_reaper_stop ] && exit 0
  sleep 120
done
