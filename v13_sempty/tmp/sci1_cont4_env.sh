# SCI-1 fourth continuation (EXPERIMENTS_SEMPY "cont4 PRE-REGISTRATION"; user YES via Lead, moved to 137,331 after the M7 lag):
# source AFTER sci1_env.sh. From cont3 milestone 9 (global 137,331): + `worked` (MetaMathQA + GSM8K train as plain worked solutions,
# no <think>, no switch) ~2 % of tokens taken from web; everything else at cont3's measured shares. Measured row calib8 (31.5M tokens,
# these weights): worked 2.16 %, b2s 1.45 %, proto4k 0.61 %, recall4k 0.30 %; synthetic loss weight 3.71 %. Indicative only (1 milestone).
export CONT_TAG=sci1_cont4
export CONT_FROM=checkpoints_v13_sempty/sci1_cont3_449e320/step_137331.pt
export CONT_TOKENS=1499922432        # 22,887 steps x 65,536: global 137,331 -> 160,218, the original stage-B end
export SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,tools,pysci,b2s,recall4k,proto4k,worked
export WEIGHTS=0.22061,0.22982,0.0506,0.07278,0.10179,0.11802,0.0224,0.02252,0.03163,0.03513,0.04647,0.01702,0.00378,0.00849,0.01896
