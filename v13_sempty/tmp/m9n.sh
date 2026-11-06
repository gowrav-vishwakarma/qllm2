#!/bin/bash
# M9 neighbourhood (EXPERIMENTS_SEMPY "s8 checkpoint jitter" rule 3; Lead GO 2026-11-06): continue M9 (cont3 step_137331, sha 218922f9)
# for 50 optimizer steps at CONSTANT LR 8e-4 (WARMUP 0) on the cont3 stream from M9's own cursor, the 6000's update size (BATCH 16 x
# SEQ 4096 = 65,536 tokens / step; here micro-batch 1 x GRAD_ACCUM 16), saving every 10 global steps (M9 + 9 / 19 / 29 / 39 / 49), then
# s8 (+ margins) on each at the CHECK's code. DRY=1: 2 steps into a scratch dir, prints the VRAM peak.
cd "$(dirname "$0")/../.." || exit 1
source v13_sempty/tmp/sci1_env.sh; source v13_sempty/tmp/sci1_cont3_env.sh
M9=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/sci1_cont3_449e320/step_137331.pt
D=checkpoints_v13_sempty/sci1_m9n_$(git rev-parse --short HEAD); STEPS=50; [ "${DRY:-0}" = 1 ] && { D=${D}_dry; STEPS=2; }
COMMON="ECHO=1 CHRONO=1 OUT_GATE=1 DELTA=1 ANSWER_W=100 SEED=43 SEQ=4096 BLEND_WARMUP=0 GEN_EVERY=0 VAL_EVERY=0 MAX_RETRIES=0 ANSWER_IN_WINDOW=1 RESET_AT_EOS=1 DECONTAM_INDEX=.cache/decontam/v1.npy"
PK=$(mktemp); ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $PK; sleep 5; done ) & WP=$!
env $COMMON $GEOM SOURCES=$SOURCES WEIGHTS=$WEIGHTS TAG=sci1_m9n CKPT_DIR=$D LR=8e-4 WARMUP=0 LR_SCHEDULE=wsd_stable CONTINUE_FROM=$M9 \
  TARGET_TOKENS=$((STEPS * 65536)) BATCH=16 GRAD_ACCUM=16 SAVE_EVERY=0 KEEP_EVERY=10 KEEP_LAST=5 bash v13_sempty/tmp_pretrain_mix.sh
kill $WP 2>/dev/null; echo "VRAM peak MB (whole card): $(sort -n $PK | tail -1)"; rm -f $PK
[ "${DRY:-0}" = 1 ] && exit 0
W=/home/gowrav/Development/qllm_sci1check; O=$PWD/logs/s8reg
R=$(pwd); for f in $D/step_*.pt; do s=$(basename $f .pt); A=$R/$f; o=$O/v13_sempty_s8reg_M9n_${s}_spread_s7000_t400.json
  [ -s $o ] || (cd $W && .venv/bin/python scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $A --output $o --suite b2 \
    --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|saved|Error"); done
echo "=== m9n DONE $(date -u +%FT%TZ)"
