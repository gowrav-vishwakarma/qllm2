#!/bin/bash
# Lead: A9 natural-variant runs on the 4090: Mamba-2 with the text library (cold, varied wordings) and the PAM PACKAGE (mind keys via key_fn, strict + NORM, varied wordings). Gate >= 6 GB free.
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
export PACKAGE_STRICT=1 PACKAGE_NORM=1
cd /home/gowrav/Development/qllm2
until [ $(( 24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )) -ge 6000 ]; do sleep 60; done
echo "start $(date -u +%FT%TZ)"
PY=/home/gowrav/Development/qllm2/.venv/bin/python; PYM=/home/gowrav/.venvs/qllm311/bin/python
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
REF=/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.mean_raw.pt
for keys in synthetic natural; do for t in kv kv1 kvupd para tool neg natq; do
  $PYM v16/bench.py --device cuda --ckpt checkpoints_v13_sempty/mbs2b_main_685de4e/best_model.pt --arch mamba2 --task $t --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 18000 --text_package --skip_state --qstyle varied --out logs/v16_textarmV_mamba2_32768_227b1f8_20261101_1020.jsonl >> logs/v16_textarmV_mamba2_32768_227b1f8_20261101_1020.log 2>&1
done; done
for keys in synthetic natural; do for t in kv kv1 kvupd para tool neg natq; do
  $PY v16/bench.py --device cuda --ckpt $B --arch pam --task $t --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 18000 --gate_tau 10 --vec_layer 8 --centre --centre_ref $REF --key_fn v14.package_fetch:key_fn --qstyle varied --out logs/v16_pkgV_b2p2_32768_227b1f8_20261101_1020.jsonl >> logs/v16_pkgV_b2p2_32768_227b1f8_20261101_1020.log 2>&1
done; done
echo TEXTARMV-GPU-DONE
