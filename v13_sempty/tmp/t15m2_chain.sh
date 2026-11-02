#!/bin/bash
# t15pam_m2 on the 6000 (EXPERIMENTS_SEMPY "t15pam_m2" pre-registration; Lead GO 2026-11-02): b2p2's exact recipe (M2 body, no
# window, c1m2a_pre stage A) on the t15 mix at 1.50B tokens, seed 43; then the t15 readouts.
cd "$(dirname "$0")/../.." || exit 1
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty; PY=.venv/bin/python; X="symbolic\|Warning\|warn"
echo "t15m2 chain: code $H in $PWD, $(date -u +%FT%TZ)"
MIX="SOURCES=dclm,fineweb,smoltalk2_mid,recall,b2 WEIGHTS=0.43,0.40,0.10,0.04,0.03 ANSWER_W=100 SEED=43 LR=1.2e-3 WARMUP=100 BATCH=18 SEQ=2048 VAL_EVERY=4883 SAVE_EVERY=200 KEEP_EVERY=0 KEEP_LAST=1 FINEWEB_LOCAL_DIR=data/fineweb-edu MAX_RETRIES=5"
SCREEN=t15m2 SCREEN_HASH=$H ARMS="main:ECHO=1,SHARE_READS=1" BASE="$MIX TARGET_TOKENS=1380003840 PAM_LAYERS=3,7,11,15 MLP_HIDDEN=2128 INIT_FROM=$C/c1m2a_pre_7ae6f1f/best_model.pt" bash v13_sempty/tmp_scratch_screen.sh >> logs/v13_sempty_t15m2_screen_${H}.out 2>&1
n=t15m2_main_$H; B=$C/$n/best_model.pt; [ -f $B ] || { echo "$n: no best_model.pt"; exit 1; }
echo "=== $n readouts $(date -u +%FT%TZ)"
$PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $B --output logs/learning_probes/v13_sempty_${n}_spread_s7000_t400.json --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096 2>&1 | grep -E "per task|free|saved"
$PY scripts/run_skillpack_probe.py --checkpoint $B --variants skillpack,contrast --alphas 0,4 --lengths 0,2048,4096,8192 --docs 100 --out logs/diag/v13_sempty_skillpack_${n}.jsonl 2>&1 | grep -v "$X" | tail -1
$PY scripts/run_skillpack_probe.py --checkpoint $B --variants dilate --dilations 1,8 --dilate_where echo --lengths 8192 --docs 50 --seed 8700 --out logs/diag/v13_sempty_clk2_echo_${n}.jsonl 2>&1 | grep -v "$X" | tail -1
for keys in synthetic natural; do PACKAGE_STRICT=1 PACKAGE_NORM=1 $PY v16/bench.py --device cuda --ckpt $B --arch pam --task tool --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 18000 --text_package --skip_state --out logs/v16_textarm_${n}_32768_${H}_tool_${keys}.jsonl 2>&1 | tail -1 | cut -c1-100; done
echo "=== t15m2 chain DONE $(date -u +%FT%TZ)"
