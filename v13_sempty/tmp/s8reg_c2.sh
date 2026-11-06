#!/bin/bash
# s8reg C2 probes (EXPERIMENTS_SEMPY "s8reg" C2): for each C2 save (global 152,600 / 153,363 / 154,126 / 154,889 / 155,652 and the
# end = latest.pt after "Training complete"), copy it from the 6000 to d_drive, then at the CHECK's code (../qllm_sci1check): s8 spread
# (aggregates carry mean_target_margin + the free-A7 accuracy), t40 16k, MC; at the end held-out NLL and the per-group weight delta
# vs M10 for all points. Pauses cx_m2 (.PAUSED + kill mindcxm2) for the whole window; Mind resumes it after Library's DIL3.
R=ubuntu@34.131.203.207; RD=Development/qllm_sci1/checkpoints_v13_sempty; RUN=${RUN:-sci1_c2_${C2H:?set C2H = the C2 run hash}}
LD=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/$RUN; mkdir -p $LD
M10=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/sci1_long_3c2bf1f/step_152590.pt
W=/home/gowrav/Development/qllm_sci1check; PY=$W/.venv/bin/python; O=/home/gowrav/Development/qllm2/logs/s8reg; mkdir -p $O
CX=/home/gowrav/Development/qllm_cx/checkpoints_v13_sempty/cx_m2_main_c17775a
echo "=== c2 probes: waiting for C2's first save $(date -u +%FT%TZ)"
until ssh -o ConnectTimeout=20 $R "ls ~/$RD/$RUN/step_152600.pt" >/dev/null 2>&1; do sleep 120; done
touch $CX/.PAUSED; tmux kill-session -t "=mindcxm2" 2>/dev/null; sleep 5
P=$(pgrep -f "v13_sempty.train .*--checkpoint_dir checkpoints_v13_sempty/cx_m2_main_c17775a( |$)" | head -1)
[ -n "$P" ] && { kill -TERM $P; sleep 30; kill -KILL $P 2>/dev/null; }
echo "=== c2 probes: cx_m2 paused (trainer ${P:-none}) $(date -u +%FT%TZ)"
fetch() {  # $1 remote name, $2 local name
  [ -s $LD/$2 ] && return 0
  until ssh -o ConnectTimeout=20 $R "ls ~/$RD/$RUN/$1" >/dev/null 2>&1; do sleep 120; done; sleep 60
  scp -q $R:$RD/$RUN/$1 $LD/$2.part && mv $LD/$2.part $LD/$2; }
probe() {  # $1 local ckpt, $2 name
  cd $W; local n=s8reg_C2_$2_$RUN f
  f=$O/v13_sempty_${n}_spread_s7000_t400.json; [ -s $f ] || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $1 --output $f --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|saved|Error"
  f=$O/v13_sempty_${n}_t40_long_behavioral.json; [ -s $f ] || $PY scripts/run_memory_behavioral.py --model-type v13_sempty --checkpoint $1 --preset baseline_real_pm --context-lengths 16384 --positions 0,0.5,1 --association-counts 1,4,8 --trials 40 --max-context 16384 --output $f 2>&1 | grep -v scored | tail -1
  f=$O/v13_sempty_mc_${n}.json; [ -s $f ] || $PY scripts/mc_eval.py --ckpt $1 --out $f 2>&1 | grep -E '^(hellaswag|arc|piqa)'; }
ALL=""
for s in 152600 153363 154126 154889 155652; do echo "=== C2 point $s $(date -u +%FT%TZ)"; fetch step_$s.pt step_$s.pt; probe $LD/step_$s.pt $s; ALL+="$LD/step_$s.pt,"; done
echo "=== C2 end: waiting for Training complete $(date -u +%FT%TZ)"
until ssh -o ConnectTimeout=20 $R "grep -q '^Training complete' \$(ls -t ~/Development/qllm_c2/logs/v13_sempty_sci1_c2_*_2026*.log | grep -v orch | head -1)" 2>/dev/null; do sleep 120; done
fetch latest.pt step_156405_end.pt; probe $LD/step_156405_end.pt 156405end; ALL+="$LD/step_156405_end.pt"
cd $W; f=$O/v13_sempty_s8reg_C2_nll_$RUN.json; [ -s $f ] || $PY /home/gowrav/Development/qllm2/v13_sempty/tmp/s8reg_d.py nll --out $f \
  --filler /home/gowrav/Development/qllm2/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt --ckpts $M10,$ALL
f=$O/v13_sempty_s8reg_C2_wdelta_$RUN.json; [ -s $f ] || $PY /home/gowrav/Development/qllm2/v13_sempty/tmp/s8reg_d.py wdelta --ref $M10 --ckpts $ALL --out $f
echo "=== s8reg C2 probes DONE $(date -u +%FT%TZ)"
