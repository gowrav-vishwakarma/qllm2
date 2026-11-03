# SCI-1 first continuation (EXPERIMENTS_SEMPY "SCI-1 first continuation"): source AFTER sci1_env.sh (it keeps GEOM / SEQ / BATCH / LR /
# KEEP_EVERY / DECAY_TOKENS and overrides the mix). From milestone 4 (global step 61,036), the rest of the planned stage B.
export CONT_TAG=sci1_cont1
export CONT_FROM=checkpoints_v13_sempty/sci1_main_59c3cf5/step_061036.pt
export CONT_TOKENS=6499991552        # exactly 99,182 steps x 65,536: global 61,036 -> 160,218, the original stage-B end
export SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,tools,pysci,b2s,recall4k,proto4k,protoc
export WEIGHTS=0.23538,0.24337,0.04951,0.07549,0.10178,0.11172,0.02436,0.02232,0.03308,0.03257,0.04891,0.00712,0.00376,0.00957,0.00108
