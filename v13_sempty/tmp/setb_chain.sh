#!/bin/bash
# Twin (c) Set B on the 6000 (EXPERIMENTS_SEMPY "Tonight's 100M / 1B twins" (c) + addenda): far protocol share {0.8, 1.6 %} x constant
# LR {1.2e-3, 6e-4}, 0.4B each, continued from wio pam main stable_final (step 13,733, 0.9B). Parent scope tuple first (validity
# C 2k >= 0.5, else STOP). Arms start when fewer than MAXRUN trainers run on the box and VRAM is free (Set A arms finishing free slots).
# Saves at absolute multiples of 1,526 steps (= +0.1B marks after 13,733); each save gets a 2k scope tuple (Library's command), then
# the step save is deleted (latest kept).   tmux new-session -d -s twsetb "bash v13_sempty/tmp/setb_chain.sh 2>&1 | tee -a logs/..."
cd "$(dirname "$0")/../.." || exit 1
H=$(git rev-parse --short HEAD); C=checkpoints_v13_sempty; PY=.venv/bin/python; MAXRUN=${MAXRUN:-3}
PARENT=$C/wio_pam_main_227f3eb/stable_final.pt
S08=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,tools,pysci,b2s,recall4k,proto4k,protoc,b2s_far,proto_far
W08=0.22341,0.23099,0.04942,0.07532,0.10153,0.11154,0.02428,0.02231,0.03302,0.03252,0.04883,0.01715,0.00379,0.00478,0.00112,0.01200,0.00800
W16=0.21948,0.22692,0.04942,0.07532,0.10153,0.11154,0.02428,0.02231,0.03302,0.03252,0.04883,0.01715,0.00379,0.00478,0.00112,0.01200,0.01600
COMMON="SOURCES=$S08 ANSWER_W=100 SEED=43 SEQ=4096 BATCH=16 GRAD_ACCUM=4 BLEND_WARMUP=0 GEN_EVERY=0 VAL_EVERY=2000 \
VAL_SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,tools,pysci,chatmt NO_WIKI_VAL=1 SAVE_EVERY=200 KEEP_EVERY=1526 KEEP_LAST=6 MAX_RETRIES=0 \
ANSWER_IN_WINDOW=1 RESET_AT_EOS=1 DECONTAM_INDEX=.cache/decontam/v1.npy PIN_HASH=$H CHRONO=1 ECHO=1 OUT_GATE=1 DELTA=1 \
PAM_LAYERS=3,7,11,15 MLP_HIDDEN=2128 SHARE_READS=1 CONTINUE_FROM=$PARENT LR_SCHEDULE=wsd_stable WARMUP=1 TARGET_TOKENS=400031744"
ARMS=("f08_lr12:WEIGHTS=$W08 LR=1.2e-3" "f16_lr12:WEIGHTS=$W16 LR=1.2e-3" "f16_lr06:WEIGHTS=$W16 LR=6e-4" "f08_lr06:WEIGHTS=$W08 LR=6e-4")
[ -f $HOME/.qllm_6000_hold ] && echo "hold flag set; waiting"
scope() {  # $1 ckpt, $2 label
  local o=logs/v14_scope_c_${2}_${H}_$(date +%Y%m%d_%H%M)
  ls logs/v14_scope_c_${2}_*.jsonl >/dev/null 2>&1 && grep -q "SCOPE DONE" logs/v14_scope_c_${2}_*.log 2>/dev/null && return 0
  until [ "$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}')" -ge 8000 ]; do sleep 60; done
  $PY v14/scope_probe.py --ckpt $1 --fullctx --lengths 2048 --docs 100 --device cuda --out $o.jsonl > $o.log 2>&1
  echo "=== scope $2: $(grep -c Traceback $o.log) tracebacks | $(grep 'L 2048' $o.log | cut -c1-200)"; }
ntrain() { pgrep -fc "v13_sempty.train .*--checkpoint_dir"; }
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
echo "set B chain: code $H, parent $PARENT, $(date -u +%FT%TZ)"
scope $PARENT parent_s13733
C2=$(grep 'L 2048' $(ls -t logs/v14_scope_c_parent_s13733_*.log | head -1) | sed -n 's/.* C +\{0,1\}\(-\{0,1\}[0-9.]*\) .*/\1/p')
echo "=== parent C 2k = $C2"; awk "BEGIN{exit !($C2 >= 0.5)}" || { echo "=== VALIDITY FAILED (C 2k < 0.5): set B STOPPED"; exit 1; }
for a in "${ARMS[@]}"; do
  n=${a%%:*}; spec=${a#*:}; D=$C/setb_${n}_$H; [ -f $D/.done ] && continue
  until [ "$(ntrain)" -lt $MAXRUN ] && [ "$(free_mb)" -ge 21800 ] && [ ! -f $HOME/.qllm_6000_hold ]; do sleep 120; done
  echo "=== start $n $(date -u +%FT%TZ)"
  ( while true; do L=$(ls -t logs/v13_sempty_setb_${n}_*.log 2>/dev/null | head -1)
      env $COMMON $spec TAG=setb_$n CKPT_DIR=$D ${L:+RESUME_LOG=$L} bash v13_sempty/tmp_pretrain_mix.sh
      L=$(ls -t logs/v13_sempty_setb_${n}_*.log | head -1); grep -qE "^Training complete\. steps=" $L && break
      echo "=== $n exited before complete, resuming in 60 s $(date -u +%FT%TZ)"; sleep 60; done
    touch $D/.done; echo "=== $n trained $(date -u +%FT%TZ)"
    for k in $(ls $D/step_*.pt | sort); do s=$(basename $k .pt); scope $k ${n}_${s#step_} && rm -f $k; done
    scope $D/latest.pt ${n}_end; echo "=== $n DONE $(date -u +%FT%TZ)" ) &
  sleep 300
done
wait; echo "=== SET B DONE $(date -u +%FT%TZ)"
