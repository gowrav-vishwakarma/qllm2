# s8reg C2 (Lead GO 2026-11-06): 0.25B decay to LR 0 from the new M10 (sci1_long_3c2bf1f step_152590) with the rule block x2
# (b2s / recall4k / proto4k doc weights doubled, the added 0.02919 taken from dclm + fineweb in proportion); everything else = cont3.
# Saves every 763 global steps (~0.05B): M10 + 10 / 773 / 1536 / 2299 / 3062 and the end (+3815). Release engineering, no claim
# (closest prior: MiniCPM / OLMo-2 anneal mixes). Source AFTER sci1_env.sh and sci1_cont3_env.sh.
export WEIGHTS=0.21653,0.22544,0.05043,0.07254,0.10145,0.11762,0.02232,0.02244,0.03152,0.03501,0.04631,0.03392,0.00754,0.01692
export DECAY_FROM=checkpoints_v13_sempty/sci1_long_3c2bf1f/step_152590.pt DECAY_TAG=sci1_c2 DECAY_TOKENS=250019840
export KEEP_EVERY_D=763 KEEP_LAST_D=8 VAL_EVERY_D=763   # a val + [mix] line at every save (outside reviewer via Lead)
