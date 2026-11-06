# s8reg C3 (Lead GO 22:2x IST 2026-11-06): C2's recipe from M9 instead of M10: 0.25B decay to LR 0 with the rule block x2, from
# cont3 step_137331 (sha256 218922f9). Saves at global multiples of 763: M9 + 3 / 766 / 1529 / 2292 / 3055, end +3815 (141,146).
# Source AFTER sci1_env.sh and sci1_cont3_env.sh. Release engineering, no claim (MiniCPM / OLMo-2 anneal mixes).
export WEIGHTS=0.21653,0.22544,0.05043,0.07254,0.10145,0.11762,0.02232,0.02244,0.03152,0.03501,0.04631,0.03392,0.00754,0.01692
export DECAY_FROM=checkpoints_v13_sempty/sci1_cont3_449e320/step_137331.pt DECAY_TAG=sci1_c3 DECAY_TOKENS=250019840
export KEEP_EVERY_D=763 KEEP_LAST_D=8 VAL_EVERY_D=763
