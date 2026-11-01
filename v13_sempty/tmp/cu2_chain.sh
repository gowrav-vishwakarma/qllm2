#!/bin/bash
# cu2 on the 4090 (EXPERIMENTS_SEMPY "cu2" pre-registration): cu2a (fld + vfy data) then cu2b (+ ACh-1), same seed
# and data, each with its futility stop at step 4883 and the pre-registered readouts.
# Run from a clean worktree at the launch hash:  tmux new-session -d -s cu2 "bash v13_sempty/tmp/cu2_chain.sh 2>&1 | tee -a logs/..."
cd "$(dirname "$0")/../.." || exit 1
export PATH=/usr/local/cuda/bin:$PATH
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty; PY=.venv/bin/python; X="symbolic\|Warning\|warn"
B2P2=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
A=/media/gowrav/d_drive/qllm2/attr
echo "cu2 chain: code $H in $PWD, $(date -u +%FT%TZ)"
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F", " '{print $1-$2}'; }
BASE="LR=1.2e-3 TARGET_TOKENS=480006144 VAL_EVERY=4883 SAVE_EVERY=500 KEEP_EVERY=0 KEEP_LAST=1 PAM_LAYERS=3,7,11,15 MLP_HIDDEN=2128 SEED=43 SOURCES=dclm,fineweb,smoltalk2_mid,recall,b2,fld,vfy WEIGHTS=0.42,0.39,0.10,0.04,0.03,0.01,0.01 INIT_FROM=$C/c1m2a_pre_7ae6f1f/best_model.pt BATCH=18 SEQ=2048 GRAD_ACCUM=3 ANSWER_IN_WINDOW=1 MAX_RETRIES=5"
futility() {   # $1 = screen name, $2 = bar at step 4883
  until L=$(ls -t logs/v13_sempty_$1_main_${H}_*.log 2>/dev/null | head -1) && [ -n "$L" ] && grep -aq "\[val @ step 4883\]" "$L"; do sleep 120; done
  v=$(grep -a "\[val @ step 4883\]" "$L" | head -1 | sed 's/.*val_loss=\([0-9.]*\).*/\1/')
  if awk -v v="$v" -v b="$2" 'BEGIN{exit !(v > b)}'; then echo "$1: FUTILITY STOP ($v > $2) $(date -u +%FT%TZ)"
    touch $C/$1_main_$H/.stop $C/$1_main_$H/.futility; pkill -f "checkpoint_dir checkpoints_v13_sempty/$1_main_$H"
  else echo "$1: futility check passed ($v <= $2) $(date -u +%FT%TZ)"; fi
}
readouts() {   # $1 = name, $2 = checkpoint
  n=$1; B=$2; mkdir -p logs/diag logs/learning_probes
  F=logs/diag/v13_sempty_fldprobe_${n}.jsonl; [ -s $F ] || $PY scripts/run_fld_probe.py --checkpoint $B --docs 100 --device cuda --out $F 2>&1 | grep -v "$X" | tail -1
  for k in natural synthetic; do V=logs/diag/v13_sempty_curprobe_${n}_verify${k}.jsonl; [ -s $V ] || $PY scripts/run_cur_probe.py --checkpoint $B --docs 200 --device cuda --keys $k --out $V 2>&1 | tail -1; done
  S=logs/diag/v13_sempty_skillpack_${n}_contrast.jsonl; [ -s $S ] || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants skillpack,contrast --alphas 0,4 --lengths 0,2048,4096 --docs 100 --mem_frac 0.25 --out $S 2>&1 | grep -v "$X" | tail -1
  T=logs/diag/v13_sempty_skillpack_${n}_tag.jsonl; [ -s $T ] || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants skillpack,contrast --alphas 0,4 --lengths 0,2048,4096 --docs 100 --mem_frac 0.25 --tag --out $T 2>&1 | grep -v "$X" | tail -1
  LP=logs/learning_probes/v13_sempty_${n}_b2_t400.json; [ -f $LP ] || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $B --output $LP --suite b2 --trials 400 2>&1 | grep -E "per task|free|saved"
  export ATTR_ROOT=$PWD
  [ -f $A/loss_${n}.npz ] || $PY v13_sempty/tmp/attr3b.py score $A/data.npz $A/loss_${n}.npz ${n}=$B 2>&1 | grep -v "$X"
  $PY v13_sempty/tmp/attr3b.py report $A/data.npz ${n} $A/loss_${n}.npz $A/loss_b2p2_ref.npz 2>&1 | grep -v "$X" | tee logs/diag/v13_sempty_attr3b_${n}_vs_b2p2.txt | grep "loss\|total"
  P=logs/diag/v13_sempty_cd3prose_${n}.json; [ -s $P ] || OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= $PY v13_sempty/tmp/cd3_plain_text.py $B $P 2>&1 | grep -v "$X" | tail -3
  echo "=== shown generations $n"; PY=$PY bash v13_sempty/tmp/show3.sh $B 2>&1 | grep -v "$X\|^W1" | tee logs/diag/v13_sempty_show3_${n}.txt | tail -12
}
for arm in cu2a cu2b; do
  until [ "$(free_mb)" -ge 16000 ]; do sleep 60; done          # measured peak 12.9 GB + 3
  if [ $arm = cu2a ]; then EXTRA_ENV=""; BAR=3.7908
  else
    EXTRA_ENV="ACH_LAMBDA=0.2 ACH_ALPHA=4.0 ACH_FILL=4096"
    VA=$(grep -a "\[val @ step 4883\]" $(ls -t logs/v13_sempty_cu2a_main_${H}_*.log | head -1) | head -1 | sed 's/.*val_loss=\([0-9.]*\).*/\1/')
    BAR=$(awk -v v="$VA" 'BEGIN{printf "%.4f", v + 0.02}')
  fi
  echo "=== $arm start $(date -u +%FT%TZ), futility bar $BAR at step 4883"
  futility $arm $BAR & FPID=$!
  SCREEN=$arm SCREEN_HASH=$H ARMS="main:ECHO=1,SHARE_READS=1" BASE="$BASE $EXTRA_ENV" bash v13_sempty/tmp_scratch_screen.sh
  kill $FPID 2>/dev/null; wait $FPID 2>/dev/null   # the watcher never outlives its arm (a crash before step 4883 cannot hang the chain)
  [ -f $C/${arm}_main_$H/.futility ] && { echo "$arm stopped by futility"; continue; }
  CK=$C/${arm}_main_$H/best_model.pt
  [ -f $CK ] || { echo "$arm: no best_model.pt"; continue; }
  echo "=== $arm readouts $(date -u +%FT%TZ)"; readouts ${arm}_main_$H $CK
  echo "=== $arm DONE (chain) $(date -u +%FT%TZ)"
done
echo "=== cu2 chain DONE $(date -u +%FT%TZ)"
