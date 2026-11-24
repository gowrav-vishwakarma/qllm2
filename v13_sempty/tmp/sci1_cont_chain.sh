#!/bin/bash
# SCI-1 continuation segment (EXPERIMENTS_SEMPY "SCI-1 first continuation"): --continue_from a FULL stage-B milestone with a fresh
# budget and the re-weighted mix (logged in the [continue] header), constant LR, full milestones every KEEP_EVERY global steps (last 4
# kept), then stable_final + the decay branch. The milestone CHECK runs on the 4090 (v13_sempty/tmp/sci1_check_local.sh), not here
# (the trainer holds ~92 GB of the 6000).
#   source v13_sempty/tmp/sci1_env.sh && source v13_sempty/tmp/sci1_cont1_env.sh && bash v13_sempty/tmp/sci1_cont_chain.sh
cd "$(dirname "$0")/../.." || exit 1
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty
: "${GEOM:?}" "${CONT_FROM:?}" "${CONT_TAG:?}" "${CONT_TOKENS:?}" "${DECAY_TOKENS:?}" "${BATCH:?}" "${GRAD_ACCUM:?}" "${LR:?}" "${KEEP_EVERY:?}" "${SOURCES:?}" "${WEIGHTS:?}"
disk_gb() { df -BG / | awk 'NR==2{gsub("G","",$4); print $4}'; }
[ "$(disk_gb)" -ge 25 ] || { echo "sci1 cont: disk $(disk_gb)G < 25, REFUSE $(date -u +%FT%TZ)"; exit 1; }
[ -s "$CONT_FROM" ] || { echo "sci1 cont: no $CONT_FROM"; exit 1; }
echo "sci1 continuation chain: code $H in $PWD, $(date -u +%FT%TZ); from $CONT_FROM; tokens $CONT_TOKENS; SOURCES=$SOURCES WEIGHTS=$WEIGHTS"
COMMON="ECHO=1 CHRONO=1 OUT_GATE=1 DELTA=1 ANSWER_W=100 SEED=43 SEQ=${SEQ:-4096} BLEND_WARMUP=0 FINEWEB_LOCAL_DIR=data/fineweb-edu GEN_EVERY=0 VAL_EVERY=2000 VAL_SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,tools,pysci,chatmt MAX_RETRIES=20 ANSWER_IN_WINDOW=1 RESET_AT_EOS=1 DECONTAM_INDEX=${DECONTAM_INDEX_C:-.cache/decontam/v1.npy}"
MAIN=${MAIN_DIR:-$C/${CONT_TAG}_$H}; DEC=$C/sci1_decay_$H
# DECAY_ONLY=1 MAIN_DIR=<finished main dir>: run only the decay branch from $MAIN/stable_final.pt (main already done, e.g. after a
# failed decay start: sft1 2026-11-24, orphan producers held the GPU).
if [ -z "${DECAY_ONLY:-}" ]; then
env $COMMON $GEOM SOURCES=$SOURCES WEIGHTS=$WEIGHTS TAG=$CONT_TAG CKPT_DIR=$MAIN PIN_HASH=$H LR=$LR WARMUP=${WARMUP_C:-500} LR_SCHEDULE=wsd_stable \
  CONTINUE_FROM=$CONT_FROM TARGET_TOKENS=$CONT_TOKENS BATCH=$BATCH GRAD_ACCUM=$GRAD_ACCUM SAVE_EVERY=500 KEEP_EVERY=$KEEP_EVERY KEEP_LAST=${KEEP_LAST:-4} \
  bash v13_sempty/tmp_pretrain_mix.sh
RC=$?; echo "=== sci1 continuation main exit=$RC $(date -u +%FT%TZ)"; [ "$RC" -eq 0 ] || exit $RC
cp $MAIN/latest.pt $MAIN/stable_final.pt
else
  [ -s $MAIN/stable_final.pt ] || { echo "sci1: DECAY_ONLY but no $MAIN/stable_final.pt"; exit 1; }
fi
echo "=== sci1 decay branch start (from $MAIN/stable_final.pt) $(date -u +%FT%TZ)"
[ "$(disk_gb)" -ge 12 ] || { echo "sci1: disk $(disk_gb)G < 12 before decay, STOP"; exit 1; }
env $COMMON $GEOM SOURCES=$SOURCES WEIGHTS=$WEIGHTS TAG=sci1_decay CKPT_DIR=$DEC PIN_HASH=$H LR=$LR LR_SCHEDULE=decay CONTINUE_FROM=$MAIN/stable_final.pt \
  TARGET_TOKENS=$DECAY_TOKENS BATCH=$BATCH GRAD_ACCUM=$GRAD_ACCUM SAVE_EVERY=500 KEEP_EVERY=0 KEEP_LAST=1 bash v13_sempty/tmp_pretrain_mix.sh
echo "=== sci1 decay exit=$? $(date -u +%FT%TZ)"; echo "=== sci1 continuation chain DONE $(date -u +%FT%TZ)"
