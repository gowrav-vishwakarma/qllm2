#!/bin/bash
# Lead for Library: A5 synonym cell (pre-registered 01:13 IST): b2p2 LENIENT package, 32k, seed 25000, 40 docs, syn + synneg. After the baseline natq rows.
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
export PACKAGE_T=-1
cd /home/gowrav/Development/qllm2
D=$(ls -t logs/v16_pkg_natq_baselines_driver_*.log | head -1)
until grep -q "PKG-NATQ-BASE-DONE" "$D"; do sleep 30; done
until [ $(( 24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )) -ge 6000 ]; do sleep 30; done
echo "start $(date -u +%FT%TZ)"
REF=/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.mean_raw.pt
PY=/home/gowrav/Development/qllm2/.venv/bin/python
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
for t in syn synneg; do
  $PY v16/bench.py --device cuda --ckpt $B --arch pam --task $t --needles 8 --lengths 32768 --docs 40 --seed 25000 --gate_tau 10 --vec_layer 8 --centre --centre_ref $REF --key_fn v14.package_fetch:key_fn --out logs/v16_pkg_syn_b2p2_32768_2738b2c_20261101_0130.jsonl >> logs/v16_pkg_syn_b2p2_32768_2738b2c_20261101_0130.log 2>&1
done
echo PKG-SYN-DONE
