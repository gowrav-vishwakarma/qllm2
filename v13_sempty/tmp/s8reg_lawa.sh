#!/bin/bash
# s8reg LAWA-lite (EXPERIMENTS_SEMPY "s8reg" LAWA-lite): per point (L1 / L2 / L3) the CHECK's s8 spread, t40 16k and MC
# (HS / ARC-e / PIQA) at the CHECK's code (../qllm_sci1check), then held-out NLL for all points + M9 + the decay.
W=/home/gowrav/Development/qllm_sci1check; PY=$W/.venv/bin/python; O=/home/gowrav/Development/qllm2/logs/s8reg; mkdir -p $O
L=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty; D=$L/s8reg_lawa; cd $W || exit 1
for p in L1 L2 L3; do B=$D/${p}_weights.pt; n=s8reg_LAWA_${p}; echo "=== LAWA $p $(date -u +%FT%TZ)"
  f=$O/v13_sempty_${n}_spread_s7000_t400.json; [ -s $f ] || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $B --output $f --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|saved|Error"
  f=$O/v13_sempty_${n}_t40_long_behavioral.json; [ -s $f ] || $PY scripts/run_memory_behavioral.py --model-type v13_sempty --checkpoint $B --preset baseline_real_pm --context-lengths 16384 --positions 0,0.5,1 --association-counts 1,4,8 --trials 40 --max-context 16384 --output $f 2>&1 | grep -v scored | tail -1
  f=$O/v13_sempty_mc_${n}.json; [ -s $f ] || $PY scripts/mc_eval.py --ckpt $B --out $f 2>&1 | grep -E '^(hellaswag|arc|piqa)'
done
f=$O/v13_sempty_s8reg_LAWA_nll.json; [ -s $f ] || $PY /home/gowrav/Development/qllm2/v13_sempty/tmp/s8reg_d.py nll --out $f --filler /home/gowrav/Development/qllm2/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt \
  --ckpts $L/sci1_cont3_449e320/c3b_step_137331_weights.pt,$L/sci1_decay_m9_8290f7d/decay_m9_library_weights.pt,$D/L1_weights.pt,$D/L2_weights.pt,$D/L3_weights.pt
echo "=== s8reg LAWA DONE $(date -u +%FT%TZ)"
# cx_m2 resumes at the end of the C3 probes (v13_sempty/tmp/s8reg_c3.sh), not here (Lead 2026-11-06: C3 GO)
