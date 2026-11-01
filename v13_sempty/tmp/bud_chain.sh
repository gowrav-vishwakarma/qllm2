#!/bin/bash
# bud on the 6000 (EXPERIMENTS_SEMPY "bud" pre-registration): the loss-weight budget check at 600M on r3c's 16nb body.
# bud0 (r3c mix) + bud1 (r3p mix) side by side, then bud2 (r3p mix, synthetic answers x20), then the readouts.
# Run from a clean worktree at the launch hash (links like qllm_cu1):  tmux new-session -d -s bud "bash v13_sempty/tmp/bud_chain.sh 2>&1 | tee -a logs/..."
cd "$(dirname "$0")/../.." || exit 1
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty; PY=.venv/bin/python
echo "bud chain: code $H in $PWD, $(date -u +%FT%TZ)"
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F", " '{print $1-$2}'; }
g=$(df -BG / | awk 'NR==2{gsub("G","",$4); print $4}'); [ "$g" -ge 8 ] || { echo "bud: disk ${g}G < 8, stop"; exit 1; }
COMMON="LR=1.2e-3 TARGET_TOKENS=480006144 WARMUP=100 VAL_EVERY=4883 SAVE_EVERY=500 KEEP_EVERY=0 KEEP_LAST=1 SEED=42 INIT_FROM=$C/r3ca_pre_38e3c29/best_model.pt BATCH=18 SEQ=2048 GRAD_ACCUM=2 FINEWEB_LOCAL_DIR=data/fineweb-edu MAX_RETRIES=5"
R3C="SOURCES=dclm,fineweb,smoltalk2_mid,recall WEIGHTS=0.44,0.41,0.10,0.05"
R3P="SOURCES=dclm,fineweb,smoltalk2_mid,recall,b2,lab,fsq,hop WEIGHTS=0.432,0.398,0.10,0.04,0.01,0.01,0.005,0.005 ANSWER_IN_WINDOW=1"
arm() { SCREEN=$1 SCREEN_HASH=$H ARMS="main:ECHO=1" BASE="$COMMON $2" bash v13_sempty/tmp_scratch_screen.sh > logs/v13_sempty_$1_screen_${H}.out 2>&1; }
until [ "$(free_mb)" -ge 56000 ]; do sleep 60; done        # two arms at ~25 GB (accum 2) + 3
echo "=== bud0 + bud1 start $(date -u +%FT%TZ)"
arm bud0 "$R3C" & P0=$!
sleep 300                                                   # stagger the two stream seeks
arm bud1 "$R3P" & P1=$!
wait $P0 $P1
echo "=== bud0 + bud1 done $(date -u +%FT%TZ): bud0 $(ls $C/bud0_main_$H/.done 2>/dev/null) bud1 $(ls $C/bud1_main_$H/.done 2>/dev/null)"
until [ "$(free_mb)" -ge 28000 ]; do sleep 60; done
echo "=== bud2 start $(date -u +%FT%TZ)"
arm bud2 "$R3P ANSWER_W_SRC=b2:20,lab:20,fsq:20,hop:20"
lp() { [ -f "$2" ] || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint "$1" --output "$2" "${@:3}" 2>&1 | grep -E "per task|free|saved"; }
for a in bud0 bud1 bud2; do
  n=${a}_main_$H; B=$C/$n/best_model.pt
  [ -f $B ] || { echo "$n: no best_model.pt"; continue; }
  echo "=== $n readouts $(date -u +%FT%TZ)"
  lp $B logs/learning_probes/v13_sempty_${n}_rules.json
  lp $B logs/learning_probes/v13_sempty_${n}_b2_t400.json --suite b2 --trials 400
  lp $B logs/learning_probes/v13_sempty_${n}_labeldiag_t200.json --suite label_diag --trials 200
  [ -s logs/diag/v13_sempty_curprobe_${n}.jsonl ] || $PY scripts/run_cur_probe.py --checkpoint $B --docs 200 --device cuda --out logs/diag/v13_sempty_curprobe_${n}.jsonl 2>&1 | tail -1
  [ -s logs/diag/v13_sempty_skillpack_${n}.jsonl ] || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants skillpack --lengths 0,2048 --docs 100 --out logs/diag/v13_sempty_skillpack_${n}.jsonl 2>&1 | tail -1
  grep -a "val @" $(ls -t logs/v13_sempty_${n}_*.log | head -1) | grep -v wiki | tail -1
done
echo "=== bud chain DONE $(date -u +%FT%TZ)"
