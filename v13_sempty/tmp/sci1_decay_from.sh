#!/bin/bash
# SCI-1 decay branch from a chosen milestone (Lead GO 2026-11-05 after M10: `worked` dropped by the keep rule, s8 0.770 -> 0.418,
# needles 0.98 -> 0.57; the decay branches from M9 = cont3 step_137331, sha256 218922f9). Same decay recipe as sci1_cont_chain.sh
# (LR 8e-4 -> 0 linear, DECAY_TOKENS 1.0B, same COMMON / GEOM), mix = the env sourced before it (cont3 for M9).
#   source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_cont3_env.sh && \
#     DECAY_FROM=checkpoints_v13_sempty/sci1_cont3_449e320/step_137331.pt DECAY_TAG=sci1_decay_m9 bash v13_sempty/tmp/sci1_decay_from.sh
cd "$(dirname "$0")/../.." || exit 1
source scripts/require_clean_tree.sh
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty
: "${GEOM:?}" "${DECAY_FROM:?}" "${DECAY_TAG:?}" "${DECAY_TOKENS:?}" "${BATCH:?}" "${GRAD_ACCUM:?}" "${LR:?}" "${SOURCES:?}" "${WEIGHTS:?}"
disk_gb() { df -BG / | awk 'NR==2{gsub("G","",$4); print $4}'; }
[ "$(disk_gb)" -ge 12 ] || { echo "sci1 decay: disk $(disk_gb)G < 12, STOP"; exit 1; }
[ -s "$DECAY_FROM" ] || { echo "sci1 decay: no $DECAY_FROM"; exit 1; }
echo "=== $DECAY_TAG start from $DECAY_FROM: code $H, $(date -u +%FT%TZ); tokens $DECAY_TOKENS; SOURCES=$SOURCES WEIGHTS=$WEIGHTS"
COMMON="ECHO=1 CHRONO=1 OUT_GATE=1 DELTA=1 ANSWER_W=100 SEED=43 SEQ=${SEQ:-4096} BLEND_WARMUP=0 FINEWEB_LOCAL_DIR=data/fineweb-edu GEN_EVERY=0 VAL_EVERY=2000 VAL_SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,tools,pysci,chatmt MAX_RETRIES=20 ANSWER_IN_WINDOW=1 RESET_AT_EOS=1 DECONTAM_INDEX=.cache/decontam/v1.npy"
env $COMMON $GEOM SOURCES=$SOURCES WEIGHTS=$WEIGHTS TAG=$DECAY_TAG CKPT_DIR=$C/${DECAY_TAG}_$H PIN_HASH=$H LR=$LR LR_SCHEDULE=decay CONTINUE_FROM=$DECAY_FROM \
  TARGET_TOKENS=$DECAY_TOKENS BATCH=$BATCH GRAD_ACCUM=$GRAD_ACCUM SAVE_EVERY=500 KEEP_EVERY=${KEEP_EVERY_D:-0} KEEP_LAST=${KEEP_LAST_D:-1} bash v13_sempty/tmp_pretrain_mix.sh
echo "=== $DECAY_TAG exit=$? $(date -u +%FT%TZ)"
