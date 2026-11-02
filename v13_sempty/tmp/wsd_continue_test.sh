#!/bin/bash
cd /home/gowrav/Development/qllm_wsdtest; export PATH=/usr/local/cuda/bin:$PATH
S=/tmp/claude-1000/-home-gowrav-Development-qllm2/99371c3c-8382-42b3-a3d3-6861632abc19/scratchpad/wsd; rm -rf $S; mkdir -p $S/A $S/B
C="ALLOW_DIRTY=1 BATCH=4 SEQ=2048 GRAD_ACCUM=1 BLEND_WARMUP=0 NO_WIKI_VAL=1 CHRONO=1 OUT_GATE=1 DELTA=1 ECHO=1 MAX_RETRIES=0 LOG_INTERVAL=5 LR=1e-3 VAL_EVERY=1000"
env $C TAG=wsdA CKPT_DIR=$S/A RESUME_LOG=$S/A/run.log SOURCES=dclm,fineweb WEIGHTS=0.5,0.5 TARGET_TOKENS=327680 WARMUP=10 LR_SCHEDULE=wsd_stable SAVE_EVERY=20 bash v13_sempty/tmp_pretrain_mix.sh > $S/A/out 2>&1
echo "A:"; grep -oE '^step [0-9]+|lr=[0-9.e-]+' $S/A/run.log | paste -sd' ' | cut -c1-300
B="TAG=wsdB CKPT_DIR=$S/B RESUME_LOG=$S/B/run.log SOURCES=dclm,fineweb,arxiv WEIGHTS=0.4,0.4,0.2 TARGET_TOKENS=163840 LR_SCHEDULE=decay CONTINUE_FROM=$S/A/latest.pt SAVE_EVERY=10"
env $C $B bash v13_sempty/tmp_pretrain_mix.sh > $S/B/out 2>&1
echo "B:"; grep -E '^\[continue\]' $S/B/run.log | cut -c1-300; grep -oE '^step [0-9]+|lr=[0-9.e-]+' $S/B/run.log | paste -sd' ' | cut -c1-300
env $C $B bash v13_sempty/tmp_pretrain_mix.sh > $S/B/out2 2>&1
echo "B rerun:"; grep -E 'already at step|\[resume\] /' $S/B/run.log | tail -2 | cut -c1-200
.venv/bin/python -c "
import torch
for n in ('A','B'):
    c=torch.load('$S/'+n+'/latest.pt', map_location='cpu', weights_only=False); a=c['args']
    print(n, 'step', c['global_step'], 'tokens', c['global_tokens'], 'cursor', {k:v for k,v in c['data_cursor']['doc_counters'].items() if '@' in k}, 'start', a.get('continue_start_step'), a.get('continue_start_tokens'))"
