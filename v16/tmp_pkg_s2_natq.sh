#!/bin/bash
# Lead: Library's S2 (strict mode, fresh seed 23000, P_free_sent primary) then natq reported rows (b2p2 lenient, T++, Mamba-2 as they are). Gate: >= 8 GB free VRAM (Mind's readouts first).
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
cd /home/gowrav/Development/qllm2
until [ $(( 24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )) -ge 8000 ]; do sleep 30; done
echo "start $(date -u +%FT%TZ)"
REF=/media/gowrav/d_drive/qllm_ckpt/v14/l1p_fwd_L8c_99e6803.mean_raw.pt
PY=/home/gowrav/Development/qllm2/.venv/bin/python
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
for keys in synthetic natural; do for t in kv para tool neg; do
  PACKAGE_STRICT=1 $PY v16/bench.py --device cuda --ckpt $B --arch pam --task $t --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 23000 --gate_tau 10 --vec_layer 8 --centre --centre_ref $REF --key_fn v14.package_fetch:key_fn --out logs/v16_pkg_s2_b2p2_32768_719d598_20261101_0106.jsonl >> logs/v16_pkg_s2_b2p2_32768_719d598_20261101_0106.log 2>&1
done; done
echo PKG-S2-DONE
for keys in synthetic natural; do
  PACKAGE_T=-1 $PY v16/bench.py --device cuda --ckpt $B --arch pam --task natq --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 24000 --gate_tau 10 --vec_layer 8 --centre --centre_ref $REF --key_fn v14.package_fetch:key_fn --out logs/v16_pkg_natq_b2p2_32768_719d598_20261101_0106.jsonl >> logs/v16_pkg_natq_b2p2_32768_719d598_20261101_0106.log 2>&1
  /home/gowrav/.venvs/qllm311/bin/python v16/bench.py --device cuda --ckpt checkpoints_v13_sempty/tfc1b_main_b788731/best_model.pt --arch transformer --task natq --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 24000 --out logs/v16_pkg_natq_transformer_32768_719d598_20261101_0106.jsonl >> logs/v16_pkg_natq_transformer_32768_719d598_20261101_0106.log 2>&1
  /home/gowrav/.venvs/qllm311/bin/python v16/bench.py --device cuda --ckpt checkpoints_v13_sempty/mbs2b_main_685de4e/best_model.pt --arch mamba2 --task natq --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 24000 --out logs/v16_pkg_natq_mamba2_32768_719d598_20261101_0106.jsonl >> logs/v16_pkg_natq_mamba2_32768_719d598_20261101_0106.log 2>&1
done
echo PKG-NATQ-DONE
