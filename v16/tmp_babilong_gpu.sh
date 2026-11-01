#!/bin/bash
# Lead: BABILong on the 4090: bodies AS THEY ARE (T++, Mamba-2, b2p2, cu2a) + the package and planner-chain columns for the PAM minds, qa1-qa5, 0k-32k, 100 docs. Waits for the exact tmux sessions w1w / a25 / kb1 to be gone and >= 7 GB free.
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
cd /home/gowrav/Development/qllm2
until ! tmux has-session -t =w1w 2>/dev/null && ! tmux has-session -t =a25 2>/dev/null && ! tmux has-session -t =kb1 2>/dev/null && [ $(( 24564 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) )) -ge 7000 ]; do sleep 120; done
echo "start $(date -u +%FT%TZ)"
PY=/home/gowrav/Development/qllm2/.venv/bin/python; PYM=/home/gowrav/.venvs/qllm311/bin/python
B=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
L=0k,1k,2k,4k,8k,16k,32k
$PY  v16/babilong_eval.py --ckpt checkpoints_v13_sempty/tfc1b_main_b788731/best_model.pt --arch transformer --lengths $L --tasks qa1,qa2,qa3,qa4,qa5 --docs 100 --device cuda --out logs/v16_babilong_asis_tfc1b_773d751_20261101_1548.jsonl >> logs/v16_babilong_asis_tfc1b_773d751_20261101_1548.log 2>&1
$PYM v16/babilong_eval.py --ckpt checkpoints_v13_sempty/mbs2b_main_685de4e/best_model.pt --arch mamba2 --lengths $L --tasks qa1,qa2,qa3,qa4,qa5 --docs 100 --device cuda --out logs/v16_babilong_asis_mbs2b_773d751_20261101_1548.jsonl >> logs/v16_babilong_asis_mbs2b_773d751_20261101_1548.log 2>&1
$PY  v16/babilong_eval.py --ckpt $B --arch pam --lengths $L --tasks qa1,qa2,qa3,qa4,qa5 --docs 100 --device cuda --package --chain --out logs/v16_babilong_full_b2p2_773d751_20261101_1548.jsonl >> logs/v16_babilong_full_b2p2_773d751_20261101_1548.log 2>&1
$PY  v16/babilong_eval.py --ckpt checkpoints_v13_sempty/cu2a_main_24b2969/best_model.pt --arch pam --lengths $L --tasks qa1,qa2,qa3,qa4,qa5 --docs 100 --device cuda --package --chain --out logs/v16_babilong_full_cu2a_773d751_20261101_1548.jsonl >> logs/v16_babilong_full_cu2a_773d751_20261101_1548.log 2>&1
echo BABILONG-GPU-DONE
