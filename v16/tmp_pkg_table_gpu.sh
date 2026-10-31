#!/bin/bash
# package table on the 4090 (Lead): waits for the baseline rows to finish, then PAM package rows with Library's fetch
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
export PACKAGE_T=-1
cd /home/gowrav/Development/qllm2
D=$(ls -t logs/v16_pkg_baselines_driver_*.log | head -1)
until grep -q "BASELINES-DONE" "$D"; do sleep 30; done
H=$(git rev-parse --short HEAD); TS=$(date +%Y%m%d_%H%M)
REF=/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.mean_raw.pt
PY=/home/gowrav/Development/qllm2/.venv/bin/python
run() { name=$1; path=$2; L=$3; docs=$4; seed=$5; shift 5
  for keys in synthetic natural; do for t in "$@"; do
    $PY v16/bench.py --device cuda --ckpt $path --arch pam --task $t --needles 8 --keys $keys --lengths $L --docs $docs --seed $seed --gate_tau 10 --vec_layer 8 --centre --centre_ref $REF --key_fn v14.package_fetch:key_fn --out logs/v16_pkg_table_${name}_${L}_${H}_${TS}.jsonl >> logs/v16_pkg_table_${name}_${L}_${H}_${TS}.log 2>&1
  done; done; }
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
C=checkpoints_v13_sempty/c1ctlb_main_7ae6f1f/best_model.pt
run b2p2 $B 32768 40 18000 kv kv1 kvupd hop2 tool para neg
run c1ctlb $C 32768 40 18000 kv kv1 kvupd hop2 tool para neg
run b2p2 $B 131072 24 18500 kv kv1 para neg tool
run c1ctlb $C 131072 24 18500 kv kv1 para neg tool
echo PKG-TABLE-DONE
