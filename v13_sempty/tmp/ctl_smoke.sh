#!/bin/bash
# Matched SCI-recipe control (Lead + user GO 08:45 IST 2026-11-04): 200-step throughput / peak-VRAM smoke of each body on the 4090
# at the planned settings (cont2 mix, SEQ 4096, BATCH 16 (4 micro-batches of 4) x 4096 = 65,536 tokens per step). One at a time.
#   bash v13_sempty/tmp/ctl_smoke.sh   -> logs/v13_sempty_ctlsmoke_<arm>_<hash>_<stamp>.log
cd "$(dirname "$0")/../.." || exit 1
H=$(git rev-parse --short HEAD); source v13_sempty/tmp/sci1_cont2_env.sh
COMMON="SOURCES=$SOURCES WEIGHTS=$WEIGHTS ANSWER_W=100 SEED=43 SEQ=4096 BATCH=16 GRAD_ACCUM=4 LR=1.2e-3 WARMUP=50 BLEND_WARMUP=0 TARGET_TOKENS=13107200 VAL_EVERY=100000 SAVE_EVERY=0 NO_WIKI_VAL=1 MAX_RETRIES=0 GEN_EVERY=0 ANSWER_IN_WINDOW=1 RESET_AT_EOS=1 DECONTAM_INDEX=.cache/decontam/v1.npy"
BASELINE="CHRONO=0 OUT_GATE=0 DELTA=0 ECHO=0"
for a in "pam:ECHO=1 CHRONO=1 OUT_GATE=1 DELTA=1 PAM_LAYERS=3,7,11,15 MLP_HIDDEN=2128 SHARE_READS=1" "tpp:ARCH=transformer $BASELINE" "mamba2:ARCH=mamba2 MAMBA_N_LAYER=17 $BASELINE"; do
  n=${a%%:*}; spec=${a#*:}; T=ctlsmoke_$n; echo "=== $T ($spec) $(date -u +%FT%TZ)"
  env $COMMON $spec TAG=$T CKPT_DIR=/tmp/$T bash v13_sempty/tmp_pretrain_mix.sh > /dev/null 2>&1
  f=$(ls -t logs/v13_sempty_${T}_${H}_*.log | head -1)
  echo "$T: $(grep -A1 '^\[params\]' $f | tail -1 | cut -c1-120) | $(grep -E '^step 200 ' $f | grep -oE '[0-9]+ tok/s \(avg [0-9]+\)|GPU [0-9.]+/[0-9.]+GB' | tr '\n' ' ') | $(grep -c Traceback $f) tracebacks"
  rm -rf /tmp/$T
done
