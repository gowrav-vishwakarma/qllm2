#!/bin/bash
# w1r on the 4090 (EXPERIMENTS_SEMPY "w1r" pre-registration, Lead GO 2026-11-01): the w1s pair again at SEED=44, then the
# pre-registered readouts (s8, clk2 8k, skill pack, text-arm tool). Run from the 508aaff worktree:
#   tmux new-session -d -s w1r "bash v13_sempty/tmp/w1r_chain.sh 2>&1 | tee -a logs/v13_sempty_w1r_chain_508aaff_<stamp>.log"
cd "$(dirname "$0")/../.." || exit 1
export PATH=/usr/local/cuda/bin:$PATH
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty; PY=.venv/bin/python; X="symbolic\|Warning\|warn"
echo "w1r chain: code $H in $PWD, $(date -u +%FT%TZ)"
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F", " '{print $1-$2}'; }
until [ "$(free_mb)" -ge 17000 ]; do sleep 60; done
BASE="LR=1.2e-3 TARGET_TOKENS=100012032 VAL_EVERY=2713 SAVE_EVERY=500 KEEP_EVERY=0 KEEP_LAST=1 PAM_LAYERS=3,7,11,15 MLP_HIDDEN=2128 SEED=44 SOURCES=dclm,fineweb,smoltalk2_mid,recall,b2 WEIGHTS=0.43,0.40,0.10,0.04,0.03 INIT_FROM=$C/c1m2a_pre_7ae6f1f/best_model.pt BATCH=18 SEQ=2048 GRAD_ACCUM=3 ANSWER_IN_WINDOW=1 MAX_RETRIES=5"
# At 508aaff the arm overrides split on commas (fixed later in 2b5937f), so SWA goes via BASE: one screen call per arm.
SCREEN=w1r SCREEN_HASH=$H ARMS="w0:ECHO=1,SHARE_READS=1" BASE="$BASE" bash v13_sempty/tmp_scratch_screen.sh
SCREEN=w1r SCREEN_HASH=$H ARMS="w128:ECHO=1,SHARE_READS=1" BASE="$BASE SWA_LAYERS=5,9,13 SWA_WINDOW=128" bash v13_sempty/tmp_scratch_screen.sh
for arm in w0 w128; do
  n=w1r_${arm}_$H; B=$C/$n/best_model.pt
  [ -f $B ] || { echo "$n: no best_model.pt"; continue; }
  echo "=== $n readouts $(date -u +%FT%TZ)"; mkdir -p logs/diag logs/learning_probes
  LP=logs/learning_probes/v13_sempty_${n}_spread_s7000_t400.json
  [ -f $LP ] || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $B --output $LP --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096 2>&1 | grep -E "per task|free|saved"
  D=logs/diag/v13_sempty_clk2_echo_${n}.jsonl; [ -s $D ] || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants dilate --dilations 1,8 --dilate_where echo --lengths 8192 --docs 50 --seed 8700 --mem_frac 0.25 --out $D 2>&1 | grep -v "$X" | tail -1
  S=logs/diag/v13_sempty_skillpack_${n}.jsonl; [ -s $S ] || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants skillpack,contrast --alphas 0,4 --lengths 0,2048,4096 --docs 100 --mem_frac 0.25 --out $S 2>&1 | grep -v "$X" | tail -1
  for keys in synthetic natural; do
    O=logs/v16_textarm_${n}_32768_${H}_tool_${keys}.jsonl
    [ -s $O ] || PACKAGE_STRICT=1 PACKAGE_NORM=1 $PY v16/bench.py --device cuda --ckpt $B --arch pam --task tool --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 18000 --text_package --skip_state --out $O 2>&1 | tail -1 | cut -c1-120
  done
  echo "=== $n DONE (chain) $(date -u +%FT%TZ)"
done
echo "=== w1r chain DONE $(date -u +%FT%TZ)"
