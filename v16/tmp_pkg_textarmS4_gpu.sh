#!/bin/bash
# Lead: Mamba-2 with the text library incl. Library's step 4 (A20), varied wordings (A9), cold read. Gate >= 5 GB free (w1s may hold ~21 GB).
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
export PACKAGE_STRICT=1 PACKAGE_NORM=1
cd /home/gowrav/Development/qllm2
until [ $(( 24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )) -ge 5000 ]; do sleep 60; done
echo "start $(date -u +%FT%TZ)"
PYM=/home/gowrav/.venvs/qllm311/bin/python
for keys in synthetic natural; do for t in kv kv1 kvupd para tool neg natq; do
  $PYM v16/bench.py --device cuda --ckpt checkpoints_v13_sempty/mbs2b_main_685de4e/best_model.pt --arch mamba2 --task $t --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 18000 --text_package --skip_state --qstyle varied --text_step4 --out logs/v16_textarmS4_mamba2_32768_9a1410d_20261101_1102.jsonl >> logs/v16_textarmS4_mamba2_32768_9a1410d_20261101_1102.log 2>&1
done; done
echo TEXTARMS4-GPU-DONE
