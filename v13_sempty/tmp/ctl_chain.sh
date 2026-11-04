#!/bin/bash
# Matched SCI-recipe control on the 4090 (EXPERIMENTS_SEMPY "ctl: matched control at the SCI-1 recipe"; Lead + user GO 08:45 IST
# 2026-11-04). PAM M2 vs Transformer++ vs Mamba-2 at ~103M each, 1.0B tokens each of the cont2 mix (same tokenizer, SEQ 4096, decontam
# index, seed 43, same stream order), wsd: 0.9B stable + 0.1B linear decay. Sequential, one body at a time.
# The trainer YIELDS the card: when a CHECK / humaneval / Library eval starts on the 4090 (python processes of scripts/run_*,
# scripts/mc_eval, scripts/sci_gen, v14/a2*, v16/humaneval) or MemAvailable < 12 GB, it is stopped (SIGTERM, saves every 200 steps)
# and resumed from its latest.pt (MAX_RETRIES restarts after signal kills: the box's CPU crashes too) once the card is free again.
#   tmux new-session -d -s mindctl "bash v13_sempty/tmp/ctl_chain.sh 2>&1 | tee -a logs/v13_sempty_ctl_chain_<hash>_<stamp>.log"
cd "$(dirname "$0")/../.." || exit 1
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty; PY=.venv/bin/python; X="symbolic\|Warning\|warn"
source v13_sempty/tmp/sci1_cont2_env.sh            # SOURCES / WEIGHTS = the cont2 mix (only these two are used)
STABLE_TOKENS=900005888; DECAY_TOKENS=100007936     # 13,733 + 1,526 steps of 65,536 = 15,259 steps = 1.0B tokens
COMMON="SOURCES=$SOURCES WEIGHTS=$WEIGHTS ANSWER_W=100 SEED=43 SEQ=4096 BATCH=16 GRAD_ACCUM=4 LR=1.2e-3 BLEND_WARMUP=0 GEN_EVERY=0 \
VAL_EVERY=2000 VAL_SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,tools,pysci,chatmt NO_WIKI_VAL=1 SAVE_EVERY=200 KEEP_EVERY=0 KEEP_LAST=1 \
MAX_RETRIES=0 ANSWER_IN_WINDOW=1 RESET_AT_EOS=1 DECONTAM_INDEX=.cache/decontam/v1.npy PIN_HASH=$H"
BASELINE="CHRONO=0 OUT_GATE=0 DELTA=0 ECHO=0"
ARMS=("pam:ECHO=1 CHRONO=1 OUT_GATE=1 DELTA=1 PAM_LAYERS=3,7,11,15 MLP_HIDDEN=2128 SHARE_READS=1"
      "tpp:ARCH=transformer $BASELINE"
      "mamba2:ARCH=mamba2 MAMBA_N_LAYER=17 $BASELINE")
echo "ctl chain: code $H in $PWD, $(date -u +%FT%TZ); SOURCES=$SOURCES WEIGHTS=$WEIGHTS"
busy() { pgrep -f '^[^ ]*python[^ ]* (-u )?(-m v14\.|([^ ]*/)?(scripts/(run_|mc_eval|sci_gen|think_eval)|v14/a2|v16/humaneval/(dump|think_eval_full|rec)|v16/bench))' >/dev/null; }
memok() { [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "${1:-12}" ]; }
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F", " '{print $1-$2}'; }
# run one segment to completion, yielding the card whenever a CHECK / eval needs it; resume = same CKPT_DIR + RESUME_LOG
segment() {  # $1 tag, $2 ckpt dir, $3 env (arm spec + schedule)
  local tag=$1 dir=$2 spec=$3 log="" n=0
  while true; do
    until ! busy && memok 20 && [ "$(free_mb)" -ge "${NEED_MB:-14000}" ]; do sleep 60; done
    [ -f $dir/.done ] && return 0
    n=$((n + 1)); echo "=== $tag run $n $(date -u +%FT%TZ)"
    ( env $COMMON $spec TAG=$tag CKPT_DIR=$dir ${log:+RESUME_LOG=$log} setsid bash v13_sempty/tmp_pretrain_mix.sh ) &
    local P=$! why=""
    sleep 90; [ -z "$log" ] && log=$(ls -t logs/v13_sempty_${tag}_${H}_*.log 2>/dev/null | head -1)
    while kill -0 $P 2>/dev/null; do
      busy && { why="yield to a CHECK / eval"; break; }
      memok 12 || { why="RAM < 12 GB"; break; }
      sleep 15
    done
    if [ -n "$why" ]; then
      echo "=== $tag: $why, stopping the trainer $(date -u +%FT%TZ)"
      pkill -TERM -f "v13_sempty.train .*--checkpoint_dir $dir" ; sleep 30; pkill -KILL -f "v13_sempty.train .*--checkpoint_dir $dir"; wait $P
      continue
    fi
    wait $P; local rc=$?
    if grep -qE "^step [0-9]+ \[([0-9]+)/\1 100%\]" "$log" 2>/dev/null || [ $rc -eq 0 ] && [ -s $dir/latest.pt ]; then
      touch $dir/.done; echo "=== $tag done rc=$rc $(date -u +%FT%TZ)"; return 0; fi
    echo "=== $tag exit rc=$rc (crash?), resuming $(date -u +%FT%TZ)"; sleep 60
  done
}
readouts() {  # $1 run dir, $2 name
  local B=$1/latest.pt n=$2; [ -f $B ] || { echo "$n: no checkpoint"; return; }
  until ! busy && memok 20; do sleep 60; done
  echo "=== $n readouts $(date -u +%FT%TZ)"; mkdir -p logs/ctl
  LP=logs/ctl/v13_sempty_${n}_spread_s7000_t400.json
  [ -f $LP ] || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $B --output $LP --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|free \(A7|saved"
  D=logs/ctl/v13_sempty_clk2_echo_${n}.jsonl
  [ -s $D ] || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants dilate --dilations 1,8 --dilate_where echo --lengths 8192,16384 --docs 50 --seed 8700 --mem_frac 0.5 --out $D 2>&1 | grep -v "$X" | tail -1
  T=logs/ctl/v13_sempty_${n}_t40_behavioral.json
  [ -f $T ] || $PY scripts/run_memory_behavioral.py --model-type v13_sempty --checkpoint $B --preset baseline_real_pm --context-lengths 2048,4096,8192,16384 --positions 0,0.5,1 --association-counts 1,4,8 --trials 40 --max-context 16384 --output $T 2>&1 | grep -v scored | tail -1
  M=logs/ctl/v13_sempty_mc_${n}.json
  [ -f $M ] || $PY scripts/mc_eval.py --ckpt $B --out $M 2>&1 | grep -E '^(hellaswag|arc|piqa)'
}
for a in "${ARMS[@]}"; do
  name=${a%%:*}; spec=${a#*:}; S=$C/ctl_${name}_main_$H; D=$C/ctl_${name}_decay_$H
  segment ctl_${name}_main $S "$spec LR_SCHEDULE=wsd_stable WARMUP=500 TARGET_TOKENS=$STABLE_TOKENS" || exit 1
  [ -s $S/stable_final.pt ] || cp $S/latest.pt $S/stable_final.pt
  segment ctl_${name}_decay $D "$spec LR_SCHEDULE=decay CONTINUE_FROM=$S/stable_final.pt TARGET_TOKENS=$DECAY_TOKENS" || exit 1
  readouts $D ctl_${name}_$H
done
# B5 (Lead + Library): the prose rule held in state, A26 G1 at 2k with the protocol-twin read, PAM vs the matched T++ on the same docs
# and row; Mamba-2 too (a28_twin --win takes several baselines, Library 2026-11-04)
B5=logs/ctl/v14_a28_ctl_pam_vs_tpp_mamba2_$H.jsonl
until ! busy && memok 20; do sleep 60; done
[ -s $B5 ] || /home/gowrav/.venvs/qllm311/bin/python v14/a28_twin.py --ckpt $C/ctl_pam_decay_$H/latest.pt --win $C/ctl_tpp_decay_$H/latest.pt,$C/ctl_mamba2_decay_$H/latest.pt \
  --wording train --lengths 2048 --docs 100 --out $B5 2>&1 | tail -3
echo "=== ctl chain DONE $(date -u +%FT%TZ)"
