#!/bin/bash
# r3p on the 6000 (EXPERIMENTS_SEMPY "r3p" pre-registration, 2026-11-01): the 3B best-package mind.
# r3c's body + schedule, stage A reused (r3ca), stage B 2.88B tokens with the mix fixed by cu1's
# verdict (MIX_SOURCES / MIX_WEIGHTS, required) and ANSWER_IN_WINDOW=1; futility stop at 1.5B
# (holdout > 3.431); then the pre-registered readouts.
# Run from a worktree at the launch hash with qllm_cu1's links (.cache .venv checkpoints_v13_sempty data):
#   MIX_SOURCES=... MIX_WEIGHTS=... tmux new-session -d -s r3p "bash v13_sempty/tmp/r3p_chain.sh 2>&1 | tee -a logs/v13_sempty_r3p_driver_<hash>_<ts>.log"
# Resumable: rerun with SCREEN_HASH=<launch hash>.
cd "$(dirname "$0")/../.." || exit 1
: "${MIX_SOURCES:?set MIX_SOURCES from the cu1 verdict}"; : "${MIX_WEIGHTS:?set MIX_WEIGHTS}"
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty; PY=.venv/bin/python; X="symbolic\|Warning\|warn"
echo "r3p chain: code $H in $PWD, mix $MIX_SOURCES = $MIX_WEIGHTS, $(date -u +%FT%TZ)"
until ! tmux has-session -t =cu1 2>/dev/null && ! tmux has-session -t =cu1n9s 2>/dev/null; do sleep 120; done   # "=" = exact name (tmux matches prefixes)
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F", " '{print $1-$2}'; }
until [ "$(free_mb)" -ge 50000 ]; do sleep 60; done           # r3c's peak 46.9 GB + 3
g=$(df -BG / | awk 'NR==2{gsub("G","",$4); print $4}'); echo "r3p: disk ${g}G, VRAM free $(free_mb) MiB, $(date -u +%FT%TZ)"; [ "$g" -ge 8 ] || { echo "r3p: disk < 8 GB, stop"; exit 1; }
# futility watcher, pre-registered: holdout at step 39060 (~1.44B stage-B tokens) > 3.431 -> stop the run
futility() {
  until L=$(ls -t logs/v13_sempty_r3p_main_${H}_*.log 2>/dev/null | head -1) && [ -n "$L" ] && grep -aq "\[val @ step 39060\]" "$L"; do sleep 300; done
  v=$(grep -a "\[val @ step 39060\]" "$L" | head -1 | sed 's/.*val_loss=\([0-9.]*\).*/\1/')
  if awk -v v="$v" 'BEGIN{exit !(v > 3.431)}'; then echo "r3p: FUTILITY STOP ($v > 3.431) $(date -u +%FT%TZ)"; tmux kill-session -t =r3p
  else echo "r3p: futility check passed ($v <= 3.431) $(date -u +%FT%TZ)"; fi
}
futility &
SCREEN=r3p SCREEN_HASH=$H ARMS="main:ECHO=1" BASE="LR=1.2e-3 TARGET_TOKENS=2880000000 WARMUP=1000 VAL_EVERY=3906 SAVE_EVERY=1000 KEEP_EVERY=39062 KEEP_LAST=1 SEED=42 INIT_FROM=$C/r3ca_pre_38e3c29/best_model.pt SOURCES=$MIX_SOURCES WEIGHTS=$MIX_WEIGHTS ANSWER_IN_WINDOW=1 FINEWEB_LOCAL_DIR=data/fineweb-edu MAX_RETRIES=3" \
  bash v13_sempty/tmp_scratch_screen.sh
B=$C/r3p_main_$H/best_model.pt
[ -f $B ] || { echo "r3p: no best_model.pt, stop"; exit 1; }
mkdir -p logs/diag logs/learning_probes
lp() { [ -f "$2" ] || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint "$1" --output "$2" "${@:3}" 2>&1 | grep -E "per task|free|saved"; }
n=r3p_main_$H
lp $B logs/learning_probes/v13_sempty_${n}_rules.json
lp $B logs/learning_probes/v13_sempty_${n}_b2_t400.json --suite b2 --trials 400
lp $B logs/learning_probes/v13_sempty_${n}_labeldiag_t200.json --suite label_diag --trials 200
for m in "$B:$n" "$C/r3cb_main_38e3c29/best_model.pt:r3cb_main_38e3c29" "$C/mix3b_transformer_answ100_e072a8f/best_model.pt:mix3b_transformer_answ100_e072a8f" "$C/mix3b_mamba2_answ100_b36e3a5/best_model.pt:mix3b_mamba2_answ100_b36e3a5"; do
  ck=${m%%:*}; nm=${m#*:}   # the s8 spread cell (R2 form), baselines as they are
  lp $ck logs/learning_probes/v13_sempty_${nm}_spread_s7000_t400.json --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096
done
SP=logs/diag/v13_sempty_skillpack_${n}.jsonl
[ -s $SP ] || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants skillpack --lengths 0,2048,4096,8192,32768 --docs 100 --mem_frac 0.2 --out $SP 2>&1 | grep -v "$X" | tail -1
CP=logs/diag/v13_sempty_curprobe_${n}.jsonl
[ -s $CP ] || $PY scripts/run_cur_probe.py --checkpoint $B --docs 200 --device cuda --out $CP 2>&1 | tail -1
echo "=== shown generations (A7)"; PY=$PY bash v13_sempty/tmp/show3.sh $B 2>&1 | grep -v "$X\|^W1" | tee logs/diag/v13_sempty_show3_${n}.txt
echo "=== r3p chain DONE $(date -u +%FT%TZ)"
