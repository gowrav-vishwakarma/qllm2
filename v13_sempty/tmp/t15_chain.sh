#!/bin/bash
# t15 on the 6000 (EXPERIMENTS_SEMPY "t15" + "w1n" pre-registrations; Lead GO 2026-11-01). The matched 1.5B-token tier: the 600M tier's
# mix exactly, each body from its own stage A, seed 43.
#   1. T++ 1.5B + Mamba-2 1.5B + w1n_n0 (16nb plain, 600M) side by side (staggered; free >= peak + 3 GB each);
#   2. w1n_n128 (16nb + SWA 5,9,13 at 128, 600M) after n0;
#   3. readouts for every finished arm;
#   4. PAM 1.5B, with the body written by Mind into $C/t15_pam_body.txt ("w128" or "plain") after the w1m + w1n verdicts.
# Run from a clean worktree at the launch hash (links like qllm_bud):
#   tmux new-session -d -s t15 "bash v13_sempty/tmp/t15_chain.sh 2>&1 | tee -a logs/v13_sempty_t15_chain_<hash>_<stamp>.log"
cd "$(dirname "$0")/../.." || exit 1
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty; PY=.venv/bin/python; X="symbolic\|Warning\|warn"
echo "t15 chain: code $H in $PWD, $(date -u +%FT%TZ)"
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F", " '{print $1-$2}'; }
disk_ok() { g=$(df -BG / | awk 'NR==2{gsub("G","",$4); print $4}'); [ "$g" -ge 6 ] || { echo "t15: disk ${g}G < 6, stop $(date -u +%FT%TZ)"; return 1; }; }
MIX="SOURCES=dclm,fineweb,smoltalk2_mid,recall,b2 WEIGHTS=0.43,0.40,0.10,0.04,0.03 ANSWER_W=100 SEED=43 LR=1.2e-3 WARMUP=100 BATCH=18 SEQ=2048 VAL_EVERY=4883 SAVE_EVERY=200 KEEP_EVERY=0 KEEP_LAST=1 FINEWEB_LOCAL_DIR=data/fineweb-edu MAX_RETRIES=5"
T15="TARGET_TOKENS=1380003840"            # stage B 1.38B + stage A 0.12B = 1.50B tokens (37,435 steps of 36,864)
T06="TARGET_TOKENS=480006144"             # the 600M tier's stage B (13,021 steps)
BASELINE="CHRONO=0 OUT_GATE=0 DELTA=0 ECHO=0"
arm() {  # $1 screen name, $2 arm spec, $3 base env
  disk_ok || return 1
  SCREEN=$1 SCREEN_HASH=$H ARMS="$2" BASE="$MIX $3" bash v13_sempty/tmp_scratch_screen.sh >> logs/v13_sempty_$1_screen_${H}.out 2>&1
}
disk_ok || exit 1
until [ "$(free_mb)" -ge 23000 ]; do sleep 60; done
echo "=== t15_tpp start $(date -u +%FT%TZ)"
arm t15tpp "main:ARCH=transformer" "$T15 $BASELINE INIT_FROM=$C/tfc1a_pre_b788731/best_model.pt" & P1=$!
sleep 300
until [ "$(free_mb)" -ge 15000 ]; do sleep 60; done
echo "=== t15_mamba start $(date -u +%FT%TZ)"
arm t15mamba "main:ARCH=mamba2,MAMBA_N_LAYER=17" "$T15 $BASELINE INIT_FROM=$C/b2ma_pre_38e3c29/best_model.pt" & P2=$!
sleep 300
until [ "$(free_mb)" -ge 28000 ]; do sleep 60; done
echo "=== w1n_n0 start $(date -u +%FT%TZ)"
arm w1n "n0:ECHO=1" "$T06 GRAD_ACCUM=2 INIT_FROM=$C/r3ca_pre_38e3c29/best_model.pt"
until [ "$(free_mb)" -ge 28000 ]; do sleep 60; done
echo "=== w1n_n128 start $(date -u +%FT%TZ)"
arm w1n "n128:ECHO=1,SWA_LAYERS=5,9,13,SWA_WINDOW=128" "$T06 GRAD_ACCUM=2 INIT_FROM=$C/r3ca_pre_38e3c29/best_model.pt"
readouts() {  # $1 run name, $2 arch
  local n=$1 B=$C/$1/best_model.pt; [ -f $B ] || { echo "$n: no best_model.pt"; return; }
  echo "=== $n readouts $(date -u +%FT%TZ)"; mkdir -p logs/diag logs/learning_probes
  LP=logs/learning_probes/v13_sempty_${n}_spread_s7000_t400.json
  [ -f $LP ] || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $B --output $LP --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096 2>&1 | grep -E "per task|free|saved"
  S=logs/diag/v13_sempty_skillpack_${n}.jsonl; [ -s $S ] || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants skillpack,contrast --alphas 0,4 --lengths 0,2048,4096,8192 --docs 100 --mem_frac 0.15 --out $S 2>&1 | grep -v "$X" | tail -1
  D=logs/diag/v13_sempty_clk2_echo_${n}.jsonl; [ -s $D ] || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants dilate --dilations 1,8 --dilate_where echo --lengths 8192 --docs 50 --seed 8700 --mem_frac 0.15 --out $D 2>&1 | grep -v "$X" | tail -1
  for keys in synthetic natural; do
    O=logs/v16_textarm_${n}_32768_${H}_tool_${keys}.jsonl
    [ -s $O ] || PACKAGE_STRICT=1 PACKAGE_NORM=1 $PY v16/bench.py --device cuda --ckpt $B --arch $2 --task tool --needles 8 --keys $keys --lengths 32768 --docs 40 --seed 18000 --text_package --skip_state --out $O 2>&1 | tail -1 | cut -c1-120
  done
  grep -a "val @" $(ls -t logs/v13_sempty_${n%_$H}_${H}_*.log 2>/dev/null | head -1) 2>/dev/null | grep -v wiki | tail -1
}
readouts w1n_n0_$H pam
readouts w1n_n128_$H pam
wait $P1 $P2
readouts t15tpp_main_$H transformer
readouts t15mamba_main_$H mamba2
echo "=== waiting for $C/t15_pam_body.txt (Mind writes w128 or plain after the w1m + w1n verdicts) $(date -u +%FT%TZ)"
until [ -s $C/t15_pam_body.txt ]; do sleep 120; done
BODY=$(cat $C/t15_pam_body.txt); echo "=== t15_pam start, body $BODY $(date -u +%FT%TZ)"
until [ "$(free_mb)" -ge 28000 ]; do sleep 60; done
if [ "$BODY" = "w128" ]; then SPEC="main:ECHO=1,SWA_LAYERS=5,9,13,SWA_WINDOW=128"; else SPEC="main:ECHO=1"; fi
arm t15pam "$SPEC" "$T15 GRAD_ACCUM=2 INIT_FROM=$C/r3ca_pre_38e3c29/best_model.pt"
readouts t15pam_main_$H pam
echo "=== t15 chain DONE $(date -u +%FT%TZ)"
