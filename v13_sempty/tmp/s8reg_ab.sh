#!/bin/bash
# s8reg arms A + B (EXPERIMENTS_SEMPY "s8reg", pre-registered 9241c98): the CHECK's s8 spread probe, each arm at the SAME code as
# its reference reading: A = ctl_pam_main stable_final in ../qllm_ctl (where the decay's 0.523 was read), B = SCI-1 step 138,500
# (M9 + re-warm) in ../qllm_sci1check (where M9's 0.770 was read).
O=/home/gowrav/Development/qllm2/logs/s8reg; mkdir -p $O
run() { cd $1 && [ -s $4 ] || $2 scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $3 --output $4 --suite b2 \
  --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|free \(A7|saved|Error"; }
echo "=== s8reg A $(date -u +%FT%TZ)"; run /home/gowrav/Development/qllm_ctl .venv/bin/python checkpoints_v13_sempty/ctl_pam_main_caa4da9/stable_final.pt $O/v13_sempty_s8reg_A_ctl_pam_stable_caa4da9_spread_s7000_t400.json
echo "=== s8reg B $(date -u +%FT%TZ)"; run /home/gowrav/Development/qllm_sci1check .venv/bin/python /media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/sci1_long_3c2bf1f/step_138500_rw.pt $O/v13_sempty_s8reg_B_sci1_step_138500_3c2bf1f_spread_s7000_t400.json
echo "=== s8reg DONE $(date -u +%FT%TZ)"
