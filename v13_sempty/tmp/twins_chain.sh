#!/bin/bash
# Twins chain (2026-11-18 night, EXPERIMENTS_SEMPY "Tonight's 100M / 1B twins" + ADDENDA): night_chain_20261118.sh's recipe, scheduler,
# yields and readouts, with the arms from the environment so the same file runs (i) the 4090 seed-44 arms one after another and (ii) ONE
# arm per instance on the 6000 (several instances side by side under the VRAM gate).
#   CHAIN_ARMS="name|spec;name|spec"   e.g. "pam_s44|CHRONO=1 SEED=44 \$PAMSPEC"   ($PAMSPEC / $BASELINE / $NOFAR expand here)
#   BOX=4090 (default: user-jobs guard + ~/.qllm_check_wants_gpu yields + probe pgrep) | BOX=6000 (only ~/.qllm_6000_hold yields;
#   other arms' trainers / readouts never count as busy there)
#   NEED_MB=<MiB> overrides the per-arm VRAM gate (6000: measured peak + 3 GB)
cd "$(dirname "$0")/../.." || exit 1
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty; PY=.venv/bin/python; X="symbolic\|Warning\|warn"
SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,tools,pysci,b2s,recall4k,proto4k,protoc,b2s_far,proto_far
WEIGHTS=0.22341,0.23099,0.04942,0.07532,0.10153,0.11154,0.02428,0.02231,0.03302,0.03252,0.04883,0.01715,0.00379,0.00478,0.00112,0.01200,0.00800
STABLE_TOKENS=900005888; DECAY_TOKENS=100007936     # 13,733 + 1,526 steps of 65,536 = 15,259 steps = 1.0B tokens
COMMON="SOURCES=$SOURCES WEIGHTS=$WEIGHTS ANSWER_W=100 SEED=43 SEQ=4096 BATCH=16 GRAD_ACCUM=4 LR=1.2e-3 BLEND_WARMUP=0 GEN_EVERY=0 \
VAL_EVERY=2000 VAL_SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,tools,pysci,chatmt NO_WIKI_VAL=1 SAVE_EVERY=200 KEEP_EVERY=0 KEEP_LAST=1 \
MAX_RETRIES=0 ANSWER_IN_WINDOW=1 RESET_AT_EOS=1 DECONTAM_INDEX=.cache/decontam/v1.npy PIN_HASH=$H"
BASELINE="CHRONO=0 OUT_GATE=0 DELTA=0 ECHO=0"
PAMSPEC="ECHO=1 OUT_GATE=1 DELTA=1 PAM_LAYERS=3,7,11,15 MLP_HIDDEN=2128 SHARE_READS=1"
NOFAR="SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,tools,pysci,b2s,recall4k,proto4k,protoc WEIGHTS=0.23324,0.24116,0.04942,0.07532,0.10153,0.11154,0.02428,0.02231,0.03302,0.03252,0.04883,0.01715,0.00379,0.00478,0.00112"   # ctl's mix (caa4da9): the far block's 2.0 % back to dclm / fineweb
ARMS=(); IFS=';' read -ra _A <<< "${CHAIN_ARMS:?set CHAIN_ARMS}"
for x in "${_A[@]}"; do n=${x%%|*}; sp=$(eval echo "\"${x#*|}\""); ARMS+=("$n:$sp"); done
echo "twins chain (BOX=${BOX:-4090}) arms: ${ARMS[*]}; code $H in $PWD, $(date -u +%FT%TZ); SOURCES=$SOURCES WEIGHTS=$WEIGHTS"
userjobs() { [ "${BOX:-4090}" = 6000 ] && return 1; nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qE 'venv-maya|voice-agent'; }   # user rule 2026-11-16: never train beside the user's jobs
# busy = the user's jobs or one of Mind's CHECK / eval probes. Library's v14 probes (~4.5-9 GB) are NOT busy: wio runs beside them on
# the VRAM gate (Lead 2026-11-16, user gave us the full card); whoever starts later waits for free memory, nobody is killed.
# ~/.qllm_check_wants_gpu (set by sci1_milestone_waiter.sh while a SCI-1 milestone CHECK waits / runs) = busy: wio yields the card.
busy() { if [ "${BOX:-4090}" = 6000 ]; then [ -f $HOME/.qllm_6000_hold ]; return; fi
  userjobs && return 0; [ -f $HOME/.qllm_check_wants_gpu ] && return 0; pgrep -f '^[^ ]*python[^ ]* (-u )?([^ ]*/)?(scripts/(run_|mc_eval|sci_gen|think_eval)|v16/humaneval/(dump|think_eval_full|rec)|v16/bench)' >/dev/null; }
# other GPU jobs of ours beside the trainer (cwd / cmdline under ~/Development/qllm*, excluding the trainer $1): their tok/s is NOT an arm speed
neighbours() { local q; for q in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do [ "$q" = "$1" ] && continue
  { readlink /proc/$q/cwd; tr '\0' ' ' </proc/$q/cmdline; } 2>/dev/null | grep -q 'Development/qllm' && echo -n "$q "; done; }
# disk gate (Lead 2026-11-17: d_drive hit 100 %, the gdn decay crashed at its save): no launch with < 10 GB free under $C
diskok() { bash scripts/require_free_gb.sh $C/ ${1:-10} 2>/dev/null; }
memok() { [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "${1:-12}" ]; }
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F", " '{print $1-$2}'; }
# gdn: smoke 2026-11-16 measured peak 11,522 MiB (nvidia-smi, solo, 25 min at the arm recipe) + 3 GB (user 4090 rule: free >= measured peak + 3 GB)
declare -A NEED=([readouts]=12000); for x in "${ARMS[@]}"; do n=${x%%:*}; case $n in gdn*) NEED[$n]=${NEED_MB:-14600};; *) NEED[$n]=${NEED_MB:-18800};; esac; done
quiet() {  # $1 MB needed: not busy, RAM ok and VRAM free for 5 consecutive minutes; gives up (return 1) after $2 minutes (0 = never)
  local ok=0 t=0; while [ $ok -lt 5 ]; do if ! busy && memok 20 && diskok 10 && [ "$(free_mb)" -ge "$1" ]; then ok=$((ok + 1)); else ok=0; fi
    t=$((t + 1)); [ "${2:-0}" -gt 0 ] && [ $t -ge $2 ] && [ $ok -lt 5 ] && return 1; sleep 60; done; }
# the arm's newest log under ANY launch hash: logs are named after HEAD at launch, not $H (2026-11-17: the glob on $H found nothing, so
# the gdn main's "Training complete" was never seen, no .done was written and the chain relaunched a finished trainer for 2 h)
tag_log() { ls -t logs/ 2>/dev/null | grep -E "^v13_sempty_${1}_[0-9a-f]{7,8}_[0-9]{8}_[0-9]{4}\.log$" | head -1 | sed 's|^|logs/|'; }
finished() { local L; L=$(tag_log $1); [ -n "$L" ] && grep -qE "^Training complete\. steps=" "$L" && [ -s $2/latest.pt ]; }   # $1 tag, $2 dir
trainer_pid() { pgrep -f "v13_sempty.train .*--checkpoint_dir $1( |$)" | head -1; }
# run one segment to completion, yielding the card whenever a CHECK / eval needs it; resume = same CKPT_DIR + RESUME_LOG
segment() {  # $1 tag, $2 ckpt dir, $3 env (arm spec + schedule), $4 arm name (VRAM gate)
  local tag=$1 dir=$2 spec=$3 arm=$4 log="" n=0
  log=$(tag_log $tag)
  while true; do
    [ -f $dir/.done ] && return 0
    local P=$(trainer_pid $dir) why="" lastnb=x dwarn=""
    if [ -n "$P" ]; then echo "=== $tag: adopting running trainer $P $(date -u +%FT%TZ)"
    else
      quiet ${NEED[$arm]} 10 || { echo "=== $tag: blocked (VRAM / busy) 10 min, trying another arm $(date -u +%FT%TZ)"; return 2; }
      n=$((n + 1)); echo "=== $tag run $n $(date -u +%FT%TZ)"
      ( env $COMMON $spec TAG=$tag CKPT_DIR=$dir ${log:+RESUME_LOG=$log} setsid bash v13_sempty/tmp_pretrain_mix.sh ) &
      sleep 90; [ -z "$log" ] && log=$(tag_log $tag)
      P=$(trainer_pid $dir)
      # a trainer that exits inside 90 s may have found the run complete ("already at step N >= N; nothing to do"): same done check
      [ -z "$P" ] && finished $tag $dir && { touch $dir/.done; echo "=== $tag done (trainer found the run complete) $(date -u +%FT%TZ)"; return 0; }
      [ -z "$P" ] && { echo "=== $tag: no trainer after 90 s"; sleep 60; continue; }
    fi
    while kill -0 $P 2>/dev/null; do
      busy && { why="yield to a CHECK / eval"; break; }
      memok 12 || { why="RAM < 12 GB"; break; }
      diskok 10 || { [ -z "$dwarn" ] && echo "=== $tag: DISK < 10 GB free under $C (saves will fail) $(date -u +%FT%TZ)"; dwarn=1; }
      nb=$(neighbours $P); [ "$nb" != "${lastnb-x}" ] && { echo "=== $tag: GPU $([ -n "$nb" ] && echo "SHARED with pids $nb(tok/s not an arm speed)" || echo SOLO) from $(date -u +%FT%TZ)"; lastnb=$nb; }
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
    if finished $tag $dir; then
      touch $dir/.done; echo "=== $tag done $(date -u +%FT%TZ)"; return 0; fi
    echo "=== $tag trainer exited before 100 % (crash?), resuming $(date -u +%FT%TZ)"; sleep 60
  done
}
# json_ok: a probe output is complete = it parses (json, or every jsonl line) and is non-empty. Probes write to <out>.part; it is moved
# into place only when the probe exited 0 AND json_ok, so a crash can never leave a file that counts as done (or that deletes a save).
json_ok() { $PY -c "
import json, sys; t = open(sys.argv[1]).read().strip(); assert t
try: json.loads(t)
except ValueError: [json.loads(l) for l in t.splitlines() if l.strip()]" "$1" 2>/dev/null; }
put() { [ "$1" -eq 0 ] && json_ok "$2.part" && mv "$2.part" "$2" || { echo "    probe for $2 failed (exit $1): not kept"; rm -f "$2.part"; }; }
readouts() {  # $1 run dir, $2 name, $3 arm
  local B=$1/latest.pt n=$2 LP D T M P; [ -f $B ] || { echo "$n: no checkpoint"; return; }   # local: D / T must not clobber the caller's
  # the arm's own interpreter: gdn (fla) / mamba2 (mamba_ssm) live in ~/.venvs/qllm_gdn, .venv has neither (Argon / Lead 2026-11-17)
  P=$PY; case $3 in gdn*|mamba2) P=$HOME/.venvs/qllm_gdn/bin/python;; esac
  quiet ${NEED[readouts]}
  echo "=== $n readouts ($P) $(date -u +%FT%TZ)"; mkdir -p logs/wio
  LP=logs/wio/v13_sempty_${n}_spread_s7000_t400.json
  [ -f $LP ] || { $P scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $B --output $LP.part --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|free \(A7|saved|Error"; put ${PIPESTATUS[0]} $LP; }
  D=logs/wio/v13_sempty_clk2_echo_${n}.jsonl
  [ -s $D ] || { rm -f $D.part; $P scripts/run_skillpack_probe.py --checkpoint $B --variants dilate --dilations 1,8 --dilate_where echo --lengths 8192,16384 --docs 50 --seed 8700 --mem_frac 0.5 --out $D.part 2>&1 | grep -v "$X" | tail -1; put ${PIPESTATUS[0]} $D; }
  T=logs/wio/v13_sempty_${n}_t40_behavioral.json
  [ -f $T ] || { $P scripts/run_memory_behavioral.py --model-type v13_sempty --checkpoint $B --preset baseline_real_pm --context-lengths 2048,4096,8192,16384 --positions 0,0.5,1 --association-counts 1,4,8 --trials 40 --max-context 16384 --output $T.part 2>&1 | grep -v scored | tail -1; put ${PIPESTATUS[0]} $T; }
  M=logs/wio/v13_sempty_mc_${n}.json
  [ -f $M ] || { $P scripts/mc_eval.py --ckpt $B --out $M.part 2>&1 | grep -E '^(hellaswag|arc|piqa)|Error'; put ${PIPESTATUS[0]} $M; }
  for k in $(ls $1/step_*.pt 2>/dev/null); do   # 3-save mean: s8 on each kept decay save, then the save is deleted (d_drive space)
    LK=logs/wio/v13_sempty_${n}_$(basename $k .pt)_spread_s7000_t400.json
    [ -f $LK ] || { $P scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $k --output $LK.part --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|saved|Error"; put ${PIPESTATUS[0]} $LK; }
    [ -s $LK ] && json_ok $LK && rm -f $k
  done
  [ -s $LP ] && [ -s $D ] && [ -s $T ] && [ -s $M ]   # all four written (an OOM beside the TTS reserve leaves one missing: retried)
}
# scheduler (v2): run whichever unfinished arm fits; an arm blocked 10 min (VRAM reserve, busy card) yields its turn
step_arm() {  # $1 arm entry; returns 0 = finished (readouts done), 1 = progressed, 2 = blocked
  local name=${1%%:*} spec=${1#*:}; local S=$C/nt_${name}_main_$H D=$C/nt_${name}_decay_$H
  if [ ! -f $S/.done ]; then segment nt_${name}_main $S "$spec LR_SCHEDULE=wsd_stable WARMUP=500 TARGET_TOKENS=$STABLE_TOKENS" $name; return $(( $? == 2 ? 2 : 1 )); fi
  [ -s $S/stable_final.pt ] || cp $S/latest.pt $S/stable_final.pt
  if [ ! -f $D/.done ]; then segment nt_${name}_decay $D "$spec LR_SCHEDULE=decay CONTINUE_FROM=$S/stable_final.pt TARGET_TOKENS=$DECAY_TOKENS KEEP_EVERY=500 KEEP_LAST=4" $name; return $(( $? == 2 ? 2 : 1 )); fi
  if [ ! -f $D/.readouts ]; then quiet ${NEED[readouts]} 10 || return 2; readouts $D nt_${name}_$H $name && touch $D/.readouts; return 1; fi
  return 0
}
while true; do
  all=1; prog=0
  # arm by arm to the end (gdn, then nochrono first: the pair Lead reads first); after any progress, back to the first unfinished arm
  # a STARTED arm that is blocked (yield to a CHECK / the user) is waited for, not skipped: the next arm must not start in its place
  # (2026-11-16: gdn yielded to the 41B CHECK and the loop moved on to nochrono)
  for a in "${ARMS[@]}"; do step_arm "$a"; r=$?; [ $r -ne 0 ] && all=0; [ $r -eq 1 ] && { prog=1; break; }
    [ $r -eq 2 ] && [ -d $C/nt_${a%%:*}_main_$H ] && break; done
  [ $all -eq 1 ] && break
  [ $prog -eq 0 ] && { echo "=== all arms blocked, waiting 5 min $(date -u +%FT%TZ)"; sleep 300; }
done
# told 2k / 8k (FAR-TOLD protocol) on each arm's decay latest: run by Library on "wio chain DONE"
echo "=== wio chain DONE $(date -u +%FT%TZ)"
