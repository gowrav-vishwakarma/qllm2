#!/bin/bash
# Lead: the 'T++ + library' arm: the SAME text-only library for every body (cold read), same docs / seeds as the package table. Gate: >= 7 GB free (beside cu2b or after it).
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
export PACKAGE_STRICT=1 PACKAGE_NORM=1
cd /home/gowrav/Development/qllm2
until [ $(( 24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )) -ge 7000 ]; do sleep 60; done
echo "start $(date -u +%FT%TZ)"
PY=/home/gowrav/Development/qllm2/.venv/bin/python; PYM=/home/gowrav/.venvs/qllm311/bin/python
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
run() { name=$1; py=$2; arch=$3; path=$4; L=$5; docs=$6; seed=$7; shift 7
  for keys in synthetic natural; do for t in "$@"; do
    $py v16/bench.py --device cuda --ckpt $path --arch $arch --task $t --needles 8 --keys $keys --lengths $L --docs $docs --seed $seed --text_package --out logs/v16_textarm_${name}_${L}_569ab7f_20261101_0948.jsonl >> logs/v16_textarm_${name}_${L}_569ab7f_20261101_0948.log 2>&1
  done; done; }
run transformer $PYM transformer checkpoints_v13_sempty/tfc1b_main_b788731/best_model.pt 32768 40 18000 kv kv1 kvupd hop2 para tool neg syn natq
run mamba2 $PYM mamba2 checkpoints_v13_sempty/mbs2b_main_685de4e/best_model.pt 32768 40 18000 kv kv1 kvupd hop2 para tool neg syn natq
run b2p2 $PY pam $B 32768 40 18000 kv kv1 kvupd hop2 para tool neg syn natq
run transformer $PYM transformer checkpoints_v13_sempty/tfc1b_main_b788731/best_model.pt 131072 24 18500 kv kv1
run b2p2 $PY pam $B 131072 24 18500 kv kv1
echo TEXTARM-DONE
