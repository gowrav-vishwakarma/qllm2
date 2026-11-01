#!/bin/bash
# Lead: the 'T++ + library' arm, CPU copy for T++ and b2p2 (cold reads only, --skip_state; the state-only rows are in the package table); Mamba-2 needs the GPU (mamba_ssm).
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
export PACKAGE_STRICT=1 PACKAGE_NORM=1 THREADS=10 OMP_NUM_THREADS=10
cd /home/gowrav/Development/qllm2
PY=/home/gowrav/Development/qllm2/.venv/bin/python
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
run() { name=$1; arch=$2; path=$3; L=$4; docs=$5; seed=$6; shift 6
  for keys in synthetic natural; do for t in "$@"; do
    nice -n 10 $PY v16/bench.py --device cpu --ckpt $path --arch $arch --task $t --needles 8 --keys $keys --lengths $L --docs $docs --seed $seed --text_package --skip_state --out logs/v16_textarm_${name}_${L}_c6fa55a_20261101_0956.jsonl >> logs/v16_textarm_${name}_${L}_c6fa55a_20261101_0956.log 2>&1
  done; done; }
run transformer transformer checkpoints_v13_sempty/tfc1b_main_b788731/best_model.pt 32768 40 18000 kv kv1 kvupd hop2 para tool neg syn natq
run b2p2 pam $B 32768 40 18000 kv kv1 kvupd hop2 para tool neg syn natq
run transformer transformer checkpoints_v13_sempty/tfc1b_main_b788731/best_model.pt 131072 24 18500 kv kv1
run b2p2 pam $B 131072 24 18500 kv kv1
echo TEXTARM-CPU-DONE
