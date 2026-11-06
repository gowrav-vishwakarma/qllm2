#!/bin/bash
# s8reg C3 probes + disk keeper (EXPERIMENTS_SEMPY "s8reg" C3): for each C3 save (global 137,340 / 138,103 / 138,866 / 139,629 /
# 140,392 and the end = latest.pt), copy it to d_drive, sha-verify against the 6000 and DELETE it there (the 6000's 25 GB floor), then,
# once the 4090 is free of Library's slot and LAWA-lite, s8 (+ margins) / t40 16k / MC at the CHECK's code; at the end held-out NLL
# for M9 + all points and the per-group weight delta vs M9; finally cx_m2 resumes.
R=ubuntu@34.131.203.207; RD=Development/qllm_sci1/checkpoints_v13_sempty; RUN=sci1_c3_${C3H:?set C3H = the C3 run hash}
LD=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/$RUN; mkdir -p $LD
M9=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/sci1_cont3_449e320/step_137331.pt
W=/home/gowrav/Development/qllm_sci1check; PY=$W/.venv/bin/python; O=/home/gowrav/Development/qllm2/logs/s8reg; mkdir -p $O
fetch() {  # $1 remote name, $2 local name: copy, sha-verify, delete on the 6000
  if [ ! -s $LD/$2 ]; then
    until ssh -o ConnectTimeout=20 $R "ls ~/$RD/$RUN/$1" >/dev/null 2>&1; do sleep 120; done; sleep 60
    scp -q $R:$RD/$RUN/$1 $LD/$2.part && mv $LD/$2.part $LD/$2; fi
  local a b; a=$(sha256sum $LD/$2 | cut -c1-64); b=$(ssh -o ConnectTimeout=20 $R "sha256sum ~/$RD/$RUN/$1 2>/dev/null | cut -c1-64")
  if [ -n "$b" ] && [ "$a" = "$b" ]; then ssh $R "rm ~/$RD/$RUN/$1" && echo "  $1 sha ok ${a:0:12}, deleted on the 6000";
  elif [ -z "$b" ]; then echo "  $1 already gone on the 6000"; else echo "  $1 SHA MISMATCH ($a vs $b): kept on both"; rm -f $LD/$2; return 1; fi; }
gpu_turn() { while tmux has-session -t "=slotdil3" 2>/dev/null || tmux has-session -t "=mindlawa" 2>/dev/null; do sleep 30; done; }
probe() {  # $1 local ckpt, $2 name
  gpu_turn; cd $W; local n=s8reg_C3_$2_$RUN f
  f=$O/v13_sempty_${n}_spread_s7000_t400.json; [ -s $f ] || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $1 --output $f --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|saved|Error"
  f=$O/v13_sempty_${n}_t40_long_behavioral.json; [ -s $f ] || $PY scripts/run_memory_behavioral.py --model-type v13_sempty --checkpoint $1 --preset baseline_real_pm --context-lengths 16384 --positions 0,0.5,1 --association-counts 1,4,8 --trials 40 --max-context 16384 --output $f 2>&1 | grep -v scored | tail -1
  f=$O/v13_sempty_mc_${n}.json; [ -s $f ] || $PY scripts/mc_eval.py --ckpt $1 --out $f 2>&1 | grep -E '^(hellaswag|arc|piqa)'; }
ALL=""
for s in 137340 138103 138866 139629 140392; do echo "=== C3 point $s $(date -u +%FT%TZ)"; fetch step_$s.pt step_$s.pt || exit 1; ALL+="$LD/step_$s.pt,"; done &
FP=$!
echo "=== C3 end: waiting for Training complete $(date -u +%FT%TZ)"
until ssh -o ConnectTimeout=20 $R "grep -q '^Training complete' \$(ls -t ~/Development/qllm_c2/logs/v13_sempty_sci1_c3_*_2026*.log 2>/dev/null | head -1)" 2>/dev/null; do sleep 120; done
wait $FP
fetch latest.pt step_141146_end.pt || exit 1
for s in 137340 138103 138866 139629 140392; do probe $LD/step_$s.pt $s; done; probe $LD/step_141146_end.pt 141146end
# A3 (pre-registered before C3 started, outside reviewer via Lead): equal-weight average of the last three saves, same reads
[ -s $LD/c3avg3_weights.pt ] || /home/gowrav/.venvs/qllm311/bin/python /home/gowrav/Development/qllm2/v13_sempty/tmp/s8reg_d.py avg \
  --ckpts $LD/step_139629.pt,$LD/step_140392.pt,$LD/step_141146_end.pt --out $LD/c3avg3_weights.pt
probe $LD/c3avg3_weights.pt avg3
ALL=$(ls $LD/step_1*.pt $LD/c3avg3_weights.pt | tr '\n' ',' | sed 's/,$//')
cd $W; f=$O/v13_sempty_s8reg_C3_nll_$RUN.json; [ -s $f ] || $PY /home/gowrav/Development/qllm2/v13_sempty/tmp/s8reg_d.py nll --out $f \
  --filler /home/gowrav/Development/qllm2/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt --ckpts $M9,$ALL
f=$O/v13_sempty_s8reg_C3_wdelta_$RUN.json; [ -s $f ] || $PY /home/gowrav/Development/qllm2/v13_sempty/tmp/s8reg_d.py wdelta --ref $M9 --ckpts $ALL --out $f
echo "=== s8reg C3 probes DONE $(date -u +%FT%TZ)"
rm -f /home/gowrav/Development/qllm_cx/checkpoints_v13_sempty/cx_m2_main_c17775a/.PAUSED
tmux new-session -d -s mindcxm2 "cd /home/gowrav/Development/qllm_cx && SCREEN_HASH=c17775a NEED_CX=16000 bash v16/complex/cx_chain_live.sh 2>&1 | tee -a logs/v16_cx_chain_c17775a.log"
echo "=== cx_m2 chain restarted $(date -u +%FT%TZ)"
