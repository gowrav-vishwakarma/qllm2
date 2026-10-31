#!/bin/bash
# Lead: Library's PACKAGE strict mode (64ea182 / d2aacdf): abstain whenever the exact organ misses; fresh seed 22000, 32k, b2p2, 40 docs, kv/para/tool/neg x syn+nat.
# Waits for lg8r's training process (pid 644174) to exit and >= 8 GB free VRAM (bench 32k peak well below 5 GB) before touching the 4090.
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
export PACKAGE_STRICT=1
cd /home/gowrav/Development/qllm2
while kill -0 644174 2>/dev/null; do sleep 30; done
until [ $(( 24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )) -ge 8000 ]; do sleep 30; done
echo "start $(date -u +%FT%TZ)"
REF=/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.mean_raw.pt
PY=/home/gowrav/Development/qllm2/.venv/bin/python
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
for keys in synthetic natural; do for t in kv para tool neg; do
  $PY v16/bench.py --device cuda --ckpt $B --arch pam --task $t --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 22000 --gate_tau 10 --vec_layer 8 --centre --centre_ref $REF --key_fn v14.package_fetch:key_fn --out logs/v16_pkg_strict_b2p2_32768_d2aacdf_20261101_0035.jsonl >> logs/v16_pkg_strict_b2p2_32768_d2aacdf_20261101_0035.log 2>&1
done; done
echo PKG-STRICT-DONE
