#!/bin/bash
# cx_m2: the complex PAM with the paper body's parts (v16/complex/model.py: delta, chrono, out_gate, sparse notebooks 3/7/11/15,
# shared reads, E1 echo, MLP width 1436 -> 102.95M: every part of the paper body) at the ctl recipe (1.0B tokens of the cont2 mix, SEQ 4096, seed 43,
# 0.9B wsd_stable + 0.1B decay), read against ctl's PAM M2 (102.88M). Pre-registration: v16/complex/EXPERIMENTS_CX.md.
# The ctl scheduler (yields to CHECK / humaneval / Library evals, resumes from latest.pt on any kill, VRAM gate). Run from the
# pinned port worktree (../qllm_cx @ the launch hash, .venv -> qllm311, the venv ctl used):
#   tmux new-session -d -s mindcxm2 "cd /home/gowrav/Development/qllm_cx && bash v16/complex/cx_chain.sh 2>&1 | tee -a logs/v16_cx_chain.log"
cd "$(dirname "$0")/../.." || exit 1
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty; PY=.venv/bin/python; X="symbolic\|Warning\|warn"
source v13_sempty/tmp/sci1_cont2_env.sh            # SOURCES / WEIGHTS = the cont2 mix (only these two are used)
STABLE_TOKENS=900005888; DECAY_TOKENS=100007936     # 13,733 + 1,526 steps of 65,536 = 15,259 steps = 1.0B tokens
COMMON="SOURCES=$SOURCES WEIGHTS=$WEIGHTS ANSWER_W=100 SEED=43 SEQ=4096 BATCH=16 GRAD_ACCUM=16 LR=1.2e-3 BLEND_WARMUP=0 GEN_EVERY=0 \
VAL_EVERY=2000 VAL_SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,tools,pysci,chatmt NO_WIKI_VAL=1 SAVE_EVERY=200 KEEP_EVERY=0 KEEP_LAST=1 \
MAX_RETRIES=0 ANSWER_IN_WINDOW=1 RESET_AT_EOS=1 DECONTAM_INDEX=.cache/decontam/v1.npy PIN_HASH=$H"
BASELINE="CHRONO=0 OUT_GATE=0 DELTA=0 ECHO=0"
ARMS=("cx_m2:PRESET=baseline DELTA=1 CHRONO=1 OUT_GATE=1 ECHO=1 PAM_LAYERS=3,7,11,15 MLP_HIDDEN=1436 SHARE_READS=1")
echo "cx chain: code $H in $PWD, $(date -u +%FT%TZ); SOURCES=$SOURCES WEIGHTS=$WEIGHTS"
busy() { pgrep -f '^[^ ]*python[^ ]* (-u )?(-m v14\.|([^ ]*/)?(scripts/(run_|mc_eval|sci_gen|think_eval)|v14/[a-z]|v16/humaneval/(dump|think_eval_full|rec)|v16/bench))' >/dev/null; }
memok() { [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "${1:-12}" ]; }
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F", " '{print $1-$2}'; }
declare -A NEED=([cx_m2]=${NEED_CX:-20000} [readouts]=12000)
quiet() {  # $1 MB needed: not busy, RAM ok and VRAM free for 5 consecutive minutes; gives up (return 1) after $2 minutes (0 = never)
  local ok=0 t=0; while [ $ok -lt 5 ]; do if ! busy && memok 20 && [ "$(free_mb)" -ge "$1" ]; then ok=$((ok + 1)); else ok=0; fi
    t=$((t + 1)); [ "${2:-0}" -gt 0 ] && [ $t -ge $2 ] && [ $ok -lt 5 ] && return 1; sleep 60; done; }
trainer_pid() { pgrep -f "v13_sempty.train .*--checkpoint_dir $1( |$)" | head -1; }
# run one segment to completion, yielding the card whenever a CHECK / eval needs it; resume = same CKPT_DIR + RESUME_LOG
segment() {  # $1 tag, $2 ckpt dir, $3 env (arm spec + schedule), $4 arm name (VRAM gate)
  local tag=$1 dir=$2 spec=$3 arm=$4 log="" n=0
  log=$(ls -t logs/v13_sempty_${tag}_${H}_*.log 2>/dev/null | head -1)
  while true; do
    [ -f $dir/.done ] && return 0
    local P=$(trainer_pid $dir) why=""
    if [ -n "$P" ]; then echo "=== $tag: adopting running trainer $P $(date -u +%FT%TZ)"
    else
      quiet ${NEED[$arm]} 10 || { echo "=== $tag: blocked (VRAM / busy) 10 min, trying another arm $(date -u +%FT%TZ)"; return 2; }
      n=$((n + 1)); echo "=== $tag run $n $(date -u +%FT%TZ)"
      ( env $COMMON $spec TAG=$tag CKPT_DIR=$dir ${log:+RESUME_LOG=$log} setsid bash v13_sempty/tmp_pretrain_mix.sh ) &
      sleep 90; [ -z "$log" ] && log=$(ls -t logs/v13_sempty_${tag}_${H}_*.log 2>/dev/null | head -1)
      P=$(trainer_pid $dir); [ -z "$P" ] && { echo "=== $tag: no trainer after 90 s"; sleep 60; continue; }
    fi
    while kill -0 $P 2>/dev/null; do
      busy && { why="yield to a CHECK / eval"; break; }
      memok 12 || { why="RAM < 12 GB"; break; }
      sleep 15
    done
    if [ -n "$why" ]; then
      echo "=== $tag: $why, stopping the trainer $(date -u +%FT%TZ)"
      kill -TERM $P; sleep 30; kill -KILL $P 2>/dev/null; sleep 5
      continue
    fi
    sleep 20   # the trainer exited by itself: done if its log shows the last step at 100 %, else a crash -> resume
    # done = the trainer's own final line (the step log prints every 50 steps, so "N/N 100%" is missing when N is not a multiple of 50:
    # the T++ main, 13,733 steps, was misread as a crash 2026-11-04 19:14 IST)
    if grep -qE "^Training complete\. steps=" "$log" 2>/dev/null && [ -s $dir/latest.pt ]; then
      touch $dir/.done; echo "=== $tag done $(date -u +%FT%TZ)"; return 0; fi
    echo "=== $tag trainer exited before 100 % (crash?), resuming $(date -u +%FT%TZ)"; sleep 60
  done
}
readouts() {  # $1 run dir, $2 name
  local B=$1/latest.pt n=$2 LP D T M; [ -f $B ] || { echo "$n: no checkpoint"; return; }   # local: D / T must not clobber the caller's
  quiet ${NEED[readouts]}
  echo "=== $n readouts $(date -u +%FT%TZ)"; mkdir -p logs/cx
  LP=logs/cx/v13_sempty_${n}_spread_s7000_t400.json
  [ -f $LP ] || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $B --output $LP --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|free \(A7|saved"
  D=logs/cx/v13_sempty_clk2_echo_${n}.jsonl
  [ -s $D ] || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants dilate --dilations 1,8 --dilate_where echo --lengths 8192,16384 --docs 50 --seed 8700 --mem_frac 0.5 --out $D 2>&1 | grep -v "$X" | tail -1
  T=logs/cx/v13_sempty_${n}_t40_behavioral.json
  [ -f $T ] || $PY scripts/run_memory_behavioral.py --model-type v13_sempty --checkpoint $B --preset baseline_real_pm --context-lengths 2048,4096,8192,16384 --positions 0,0.5,1 --association-counts 1,4,8 --trials 40 --max-context 16384 --output $T 2>&1 | grep -v scored | tail -1
  M=logs/cx/v13_sempty_mc_${n}.json
  [ -f $M ] || $PY scripts/mc_eval.py --ckpt $B --out $M 2>&1 | grep -E '^(hellaswag|arc|piqa)'
  [ -s $LP ] && [ -s $D ] && [ -s $T ] && [ -s $M ]   # all four written (an OOM beside the TTS reserve leaves one missing: retried)
}
# scheduler (v2): run whichever unfinished arm fits; an arm blocked 10 min (VRAM reserve, busy card) yields its turn
step_arm() {  # $1 arm entry; returns 0 = finished (readouts done), 1 = progressed, 2 = blocked
  local name=${1%%:*} spec=${1#*:}; local S=$C/${name}_main_$H D=$C/${name}_decay_$H
  if [ ! -f $S/.done ]; then segment ${name}_main $S "$spec LR_SCHEDULE=wsd_stable WARMUP=500 TARGET_TOKENS=$STABLE_TOKENS" $name; return $(( $? == 2 ? 2 : 1 )); fi
  [ -s $S/stable_final.pt ] || cp $S/latest.pt $S/stable_final.pt
  if [ ! -f $D/.done ]; then segment ${name}_decay $D "$spec LR_SCHEDULE=decay CONTINUE_FROM=$S/stable_final.pt TARGET_TOKENS=$DECAY_TOKENS" $name; return $(( $? == 2 ? 2 : 1 )); fi
  if [ ! -f $D/.readouts ]; then quiet ${NEED[readouts]} 10 || return 2; readouts $D ${name}_$H && touch $D/.readouts; return 1; fi
  return 0
}
while true; do
  all=1; prog=0
  for a in "${ARMS[@]}"; do step_arm "$a"; r=$?; [ $r -ne 0 ] && all=0; [ $r -eq 1 ] && prog=1; done
  [ $all -eq 1 ] && break
  [ $prog -eq 0 ] && { echo "=== all arms blocked, waiting 5 min $(date -u +%FT%TZ)"; sleep 300; }
done
# B5: the told rule (A26 G1 2k twin read) on cx_m2; the package fetch path may not support the complex state: a failure is reported
quiet ${NEED[readouts]}
B5=logs/cx/v14_a28_cx_m2_$H.jsonl
[ -s $B5 ] || /home/gowrav/.venvs/qllm311/bin/python v14/a28_twin.py --ckpt $C/cx_m2_decay_$H/latest.pt --wording train --lengths 2048 --docs 100 --out $B5 2>&1 | tail -3
echo "=== cx chain DONE $(date -u +%FT%TZ)"
