#!/bin/bash
# SCI-1 size smoke (EXPERIMENTS_SEMPY "SCI-1"): each geometry alone on the 6000 for 200 steps at the run's batch, so the size is
# chosen by MEASURED tok/s and peak VRAM (4 days -> 8-10B tokens; <= ~80 GB to leave room for the milestone CHECK).
#   BODY=m2|16nb bash v13_sempty/tmp/sci1_size_smoke.sh   -> logs/v13_sempty_sci1_smoke_<geom>_<hash>.log
cd "$(dirname "$0")/../.." || exit 1
H=$(git rev-parse --short HEAD); C=checkpoints_v13_sempty; BODY=${BODY:?m2 or 16nb}
SRC="SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,smoltalk2_mid,chatmt,tools,think,pysci,recall4k,b2s,proto WEIGHTS=0.24,0.24,0.049,0.085,0.10,0.11,0.023,0.03,0.035,0.022,0.047,0.004,0.01,0.004"
for g in "896 24 7 2048 32" "896 24 7 4096 16" "1024 24 8 2048 32" "1024 24 8 4096 16" "1152 20 9 2048 32" "1152 20 9 4096 16"; do
  set -- $g; D=$1; L=$2; NH=$3; SQ=$4; BS=$5; MLP=$(( (D * 2128 / 588 + 4) / 8 * 8 )); PL=$(seq -s, 3 4 $((L - 1)))
  if [ "$BODY" = m2 ]; then GEOM="DIM=$D N_LAYERS=$L N_HEADS=$NH HEAD_DIM=128 MLP_HIDDEN=$MLP PAM_LAYERS=$PL SHARE_READS=1"
  else GEOM="DIM=$D N_LAYERS=$L N_HEADS=$NH HEAD_DIM=128"; fi
  T=sci1smoke_${BODY}_${D}x${L}_s${SQ}; echo "=== $T ($GEOM) $(date -u +%FT%TZ)"
  env $SRC $GEOM TAG=$T CKPT_DIR=/tmp/$T ECHO=1 CHRONO=1 OUT_GATE=1 DELTA=1 LR=8e-4 WARMUP=50 BATCH=$BS GRAD_ACCUM=2 SEQ=$SQ \
    TARGET_TOKENS=13107200 BLEND_WARMUP=0 VAL_EVERY=100000 SAVE_EVERY=0 NO_WIKI_VAL=1 FINEWEB_LOCAL_DIR=data/fineweb-edu MAX_RETRIES=0 \
    bash v13_sempty/tmp_pretrain_mix.sh > /dev/null 2>&1
  f=$(ls -t logs/v13_sempty_${T}_${H}_*.log | head -1)
  echo "$T: params $(grep -A1 '^\[params\]' $f | tail -1) | $(grep -E '^step 200 ' $f | grep -oE 'avg [0-9]+ tok/s|GPU [0-9.]+/[0-9.]+GB')"
  rm -rf /tmp/$T
done
