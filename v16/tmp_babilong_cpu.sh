#!/bin/bash
# Lead: BABILong package column on CPU (0k-2k; qa1-qa5, 100 docs) for b2p2 and cu2a; the as-is bodies + 4k-32k go to the 4090 after the sweep / A25.
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
export THREADS=10 OMP_NUM_THREADS=10
cd /home/gowrav/Development/qllm2
PY=/home/gowrav/Development/qllm2/.venv/bin/python
for spec in "b2p2|/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt" "cu2a|checkpoints_v13_sempty/cu2a_main_24b2969/best_model.pt"; do name=${spec%%|*}; ck=${spec#*|}
  nice -n 10 $PY v16/babilong_eval.py --ckpt $ck --arch pam --lengths 0k,1k,2k --tasks qa1,qa2,qa3,qa4,qa5 --docs 100 --device cpu --package --skip_asis --out logs/v16_babilong_pkg_${name}_608ebe0_20261101_1542.jsonl >> logs/v16_babilong_pkg_${name}_608ebe0_20261101_1542.log 2>&1
done
echo BABILONG-CPU-DONE
