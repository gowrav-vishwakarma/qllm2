# SCI-1 long run (v16/PLAN_LONGRUN.md R1-R7; Lead GO 2026-11-06 after M10): from M9 = cont3 step_137331 (9.0B, sha256 218922f9) on
# the cont3 mix (the one that produced M9; cont4's worked failed its keep rule). Open-ended: budget 100B (stopped / released by R4 / R6
# long before); milestones every 1B (KEEP_EVERY 15,259 steps), KEEP_LAST 2 on the 6000 (PLAN_LONGRUN disk amendment). The chat ramp
# enters at a later mix switch once its code + R5 audit are pre-registered. Source AFTER sci1_env.sh and sci1_cont3_env.sh.
export CONT_TAG=sci1_long
export CONT_FROM=checkpoints_v13_sempty/sci1_cont3_449e320/step_137331.pt
export CONT_TOKENS=99999940608     # 1525878 steps x 65,536
export KEEP_LAST=2
