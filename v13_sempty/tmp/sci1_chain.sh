#!/bin/bash
# SCI-1 on the 6000 (EXPERIMENTS_SEMPY "SCI-1" pre-registration; user GO via Lead). Continuable warmup-stable-decay:
#   main line  = LR_SCHEDULE wsd_stable (constant after warmup); FULL milestone step_*.pt every KEEP_EVERY steps (keep the last 4)
#                = the continuation points; latest.pt every 500 steps; crash -> the launcher auto-resumes (MAX_RETRIES);
#   CHECK      = v13_sempty/tmp/sci1_check.sh on every new milestone (background watcher), logged to logs/v13_sempty_sci1_checks_<hash>.log;
#   decay      = a separate short branch, --continue_from the final stable state, LR linear to 0; its best_model.pt = the release.
# Inputs (env): GEOM (e.g. "DIM=1024 N_LAYERS=24 N_HEADS=8 HEAD_DIM=128 MLP_HIDDEN=3704 PAM_LAYERS=3,7,11,15,19,23 SHARE_READS=1"),
#   MAIN_TOKENS, DECAY_TOKENS, BATCH, GRAD_ACCUM, LR, KEEP_EVERY, WEIGHTS / SOURCES (the pre-registered mix).
#   tmux new-session -d -s sci1 "bash v13_sempty/tmp/sci1_chain.sh 2>&1 | tee -a logs/v13_sempty_sci1_chain_<hash>_<stamp>.log"
cd "$(dirname "$0")/../.." || exit 1
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty
: "${GEOM:?}" "${MAIN_TOKENS:?}" "${DECAY_TOKENS:?}" "${BATCH:?}" "${GRAD_ACCUM:?}" "${LR:?}" "${KEEP_EVERY:?}" "${SOURCES:?}" "${WEIGHTS:?}"
disk_gb() { df -BG / | awk 'NR==2{gsub("G","",$4); print $4}'; }
[ "$(disk_gb)" -ge 40 ] || { echo "sci1: disk $(disk_gb)G < 40, REFUSE $(date -u +%FT%TZ)"; exit 1; }
echo "sci1 chain: code $H in $PWD, $(date -u +%FT%TZ); GEOM=$GEOM MAIN=$MAIN_TOKENS DECAY=$DECAY_TOKENS BATCH=$BATCH x$GRAD_ACCUM LR=$LR KEEP_EVERY=$KEEP_EVERY"
COMMON="ECHO=1 CHRONO=1 OUT_GATE=1 DELTA=1 ANSWER_W=100 SEED=43 SEQ=2048 BLEND_WARMUP=0 FINEWEB_LOCAL_DIR=data/fineweb-edu GEN_EVERY=0 VAL_EVERY=2000 VAL_SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,tools,think,pysci MAX_RETRIES=20 SOURCES=$SOURCES WEIGHTS=$WEIGHTS"
MAIN=$C/sci1_main_$H; DEC=$C/sci1_decay_$H; CHK=logs/v13_sempty_sci1_checks_$H.log
check_loop() {  # CHECK each new milestone once (before KEEP_LAST prunes it: milestones are ~1B tokens apart)
  while true; do
    for f in $(ls $MAIN/step_*.pt 2>/dev/null); do m=$(basename $f .pt); [ -f $MAIN/.checked_$m ] && continue
      sleep 60; bash v13_sempty/tmp/sci1_check.sh $f sci1_${m}_$H >> $CHK 2>&1; touch $MAIN/.checked_$m; done
    [ -f $MAIN/.done ] && break; sleep 600
  done
}
check_loop & CL=$!
env $COMMON $GEOM TAG=sci1_main CKPT_DIR=$MAIN PIN_HASH=$H LR=$LR WARMUP=2000 LR_SCHEDULE=wsd_stable TARGET_TOKENS=$MAIN_TOKENS \
  BATCH=$BATCH GRAD_ACCUM=$GRAD_ACCUM SAVE_EVERY=500 KEEP_EVERY=$KEEP_EVERY KEEP_LAST=4 bash v13_sempty/tmp_pretrain_mix.sh
RC=$?; echo "=== sci1 main line exit=$RC $(date -u +%FT%TZ)"; [ "$RC" -eq 0 ] || { kill $CL; exit $RC; }
cp $MAIN/latest.pt $MAIN/stable_final.pt && touch $MAIN/.done; wait $CL
echo "=== sci1 decay branch start (from $MAIN/stable_final.pt) $(date -u +%FT%TZ)"
[ "$(disk_gb)" -ge 12 ] || { echo "sci1: disk $(disk_gb)G < 12 before decay, STOP"; exit 1; }
env $COMMON $GEOM TAG=sci1_decay CKPT_DIR=$DEC PIN_HASH=$H LR=$LR LR_SCHEDULE=decay CONTINUE_FROM=$MAIN/stable_final.pt \
  TARGET_TOKENS=$DECAY_TOKENS BATCH=$BATCH GRAD_ACCUM=$GRAD_ACCUM SAVE_EVERY=500 KEEP_EVERY=0 KEEP_LAST=1 bash v13_sempty/tmp_pretrain_mix.sh
echo "=== sci1 decay exit=$? $(date -u +%FT%TZ)"
bash v13_sempty/tmp/sci1_check.sh $DEC/best_model.pt sci1_release_$H >> $CHK 2>&1
echo "=== sci1 chain DONE $(date -u +%FT%TZ)"
