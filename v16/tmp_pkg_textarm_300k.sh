#!/bin/bash
# Lead (user: long chats at 128k-300k): the text-only arm at 300k on CPU (cold reads, --skip_state), b2p2 + T++, kv / kv1 / kvupd / para x syn / nat, 20 docs, seed 18500; template wording (A9 varied as a second pass).
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
export PACKAGE_STRICT=1 PACKAGE_NORM=1 THREADS=10 OMP_NUM_THREADS=10
cd /home/gowrav/Development/qllm2
PY=/home/gowrav/Development/qllm2/.venv/bin/python
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
for qs in template varied; do for keys in synthetic natural; do for t in kv kv1 kvupd para; do
  for spec in "b2p2|pam|$B" "transformer|transformer|checkpoints_v13_sempty/tfc1b_main_b788731/best_model.pt"; do name=${spec%%|*}; r=${spec#*|}; arch=${r%%|*}; ck=${r#*|}
    nice -n 10 $PY v16/bench.py --device cpu --ckpt $ck --arch $arch --task $t --needles 8 --keys $keys --lengths 300000 --docs 20 --seed 18500 --text_package --skip_state --text_step4 --qstyle $qs --out logs/v16_textarm300k_${name}_${qs}_e41d8f3_20261101_1502.jsonl >> logs/v16_textarm300k_${name}_${qs}_e41d8f3_20261101_1502.log 2>&1
  done
done; done; done
echo TEXTARM300K-DONE
