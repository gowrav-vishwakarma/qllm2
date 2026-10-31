#!/bin/bash
# package-table baseline rows (Mamba-2, T++ as they are) on the 4090, same seeds as the package rows (Lead)
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
cd /home/gowrav/Development/qllm2
H=$(git rev-parse --short HEAD); TS=$(date +%Y%m%d_%H%M)
PY=/home/gowrav/.venvs/qllm311/bin/python
for CK in mamba2:checkpoints_v13_sempty/mbs2b_main_685de4e/best_model.pt transformer:checkpoints_v13_sempty/tfc1b_main_b788731/best_model.pt; do
  arch=${CK%%:*}; path=${CK#*:}
  for keys in synthetic natural; do for t in kv kv1 kvupd hop2 tool para neg; do
    $PY v16/bench.py --device cuda --ckpt $path --arch $arch --task $t --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 18000 --out logs/v16_pkg_table_${arch}_32768_${H}_${TS}.jsonl >> logs/v16_pkg_table_${arch}_32768_${H}_${TS}.log 2>&1
  done; done
done
echo BASELINES-DONE
