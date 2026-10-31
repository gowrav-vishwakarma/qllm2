#!/bin/bash
# Lead: GPU copy of the 128k fill (latest-wins, two-hop; seed 18500, 24 docs, both key types): runs after the strict job frees the 4090; the CPU copy (~80 s/doc) is killed once this lands.
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
export PACKAGE_T=-1
cd /home/gowrav/Development/qllm2
D=$(ls -t logs/v16_pkg_strict_driver_*.log | head -1)
until grep -q "PKG-STRICT-DONE" "$D"; do sleep 30; done
until [ $(( 24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )) -ge 12500 ]; do sleep 30; done
echo "start $(date -u +%FT%TZ)"
REF=/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.mean_raw.pt
PY=/home/gowrav/Development/qllm2/.venv/bin/python
run() { name=$1; path=$2; L=$3; docs=$4; seed=$5; shift 5
  for keys in synthetic natural; do for t in "$@"; do
    $PY v16/bench.py --device cuda --ckpt $path --arch pam --task $t --needles 8 --keys $keys --lengths $L --docs $docs --seed $seed --gate_tau 10 --vec_layer 8 --centre --centre_ref $REF --key_fn v14.package_fetch:key_fn --out logs/v16_pkg_table_${name}_${L}_fill_d2aacdf_20261101_0036.jsonl >> logs/v16_pkg_table_${name}_${L}_fill_d2aacdf_20261101_0036.log 2>&1
  done; done; }
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
C=checkpoints_v13_sempty/c1ctlb_main_7ae6f1f/best_model.pt
run b2p2 $B 131072 24 18500 kvupd hop2
run c1ctlb $C 131072 24 18500 kvupd hop2
echo PKG-FILL-GPU-DONE
