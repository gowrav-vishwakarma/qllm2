#!/bin/bash
# Lead: natq rows for T++ and Mamba-2 as they are (first attempt OOMed beside n11rf); after A4, with >= 12 GB free.
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
cd /home/gowrav/Development/qllm2
D=$(ls -t logs/v16_pkg_a4norm_driver_*.log | head -1)
until grep -q "PKG-A4NORM-DONE" "$D"; do sleep 30; done
until [ $(( 24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )) -ge 12000 ]; do sleep 30; done
echo "start $(date -u +%FT%TZ)"
PY=/home/gowrav/.venvs/qllm311/bin/python
for keys in synthetic natural; do
  $PY v16/bench.py --device cuda --ckpt checkpoints_v13_sempty/tfc1b_main_b788731/best_model.pt --arch transformer --task natq --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 24000 --out logs/v16_pkg_natq_transformer_32768_55a6856_20261101_0121.jsonl >> logs/v16_pkg_natq_transformer_32768_55a6856_20261101_0121.log 2>&1
  $PY v16/bench.py --device cuda --ckpt checkpoints_v13_sempty/mbs2b_main_685de4e/best_model.pt --arch mamba2 --task natq --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 24000 --out logs/v16_pkg_natq_mamba2_32768_55a6856_20261101_0121.jsonl >> logs/v16_pkg_natq_mamba2_32768_55a6856_20261101_0121.log 2>&1
done
echo PKG-NATQ-BASE-DONE
