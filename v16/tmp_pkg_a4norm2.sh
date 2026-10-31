#!/bin/bash
# Lead for Library: A4 (plumbing, pre-registered 01:04 IST): normalised-word organ inside the session (PACKAGE_NORM=1) + strict; b2p2, 32k, seed 24000, 40 docs, kv/para/tool/neg x syn+nat. Runs after natq.
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
cd /home/gowrav/Development/qllm2
D=$(ls -t logs/v16_pkg_syn_driver_*.log | head -1)
until grep -q "PKG-SYN-DONE" "$D"; do sleep 30; done
until [ $(( 24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )) -ge 6000 ]; do sleep 30; done
echo "start $(date -u +%FT%TZ)"
REF=/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.mean_raw.pt
PY=/home/gowrav/Development/qllm2/.venv/bin/python
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
for keys in synthetic natural; do for t in kv para tool neg; do
  PACKAGE_STRICT=1 PACKAGE_NORM=1 $PY v16/bench.py --device cuda --ckpt $B --arch pam --task $t --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 27000 --gate_tau 10 --vec_layer 8 --centre --centre_ref $REF --key_fn v14.package_fetch:key_fn --out logs/v16_pkg_a4norm2_b2p2_32768_b886a03_20261101_0137.jsonl >> logs/v16_pkg_a4norm2_b2p2_32768_b886a03_20261101_0137.log 2>&1
done; done
echo PKG-A4NORM2-DONE
