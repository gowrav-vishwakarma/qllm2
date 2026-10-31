#!/bin/bash
# Lead: the two 128k cells missing from the package table (latest-wins, two-hop), CPU so lg8r keeps the 4090; same seed 18500, 24 docs, both key types
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
export PACKAGE_T=-1 THREADS=12 OMP_NUM_THREADS=12
cd /home/gowrav/Development/qllm2
REF=/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.mean_raw.pt
PY=/home/gowrav/Development/qllm2/.venv/bin/python
run() { name=$1; path=$2; L=$3; docs=$4; seed=$5; shift 5
  for keys in synthetic natural; do for t in "$@"; do
    nice -n 10 $PY v16/bench.py --device cpu --ckpt $path --arch pam --task $t --needles 8 --keys $keys --lengths $L --docs $docs --seed $seed --gate_tau 10 --vec_layer 8 --centre --centre_ref $REF --key_fn v14.package_fetch:key_fn --out logs/v16_pkg_table_${name}_${L}_fill_a7e5d5f_20261101_0032.jsonl >> logs/v16_pkg_table_${name}_${L}_fill_a7e5d5f_20261101_0032.log 2>&1
  done; done; }
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
C=checkpoints_v13_sempty/c1ctlb_main_7ae6f1f/best_model.pt
run b2p2 $B 131072 24 18500 kvupd hop2
run c1ctlb $C 131072 24 18500 kvupd hop2
echo PKG-FILL-DONE
