# SCI-1 launch values (EXPERIMENTS_SEMPY "SCI-1" pre-registration, f11df39). Sourced by the launch command:
#   source v13_sempty/tmp/sci1_env.sh && bash v13_sempty/tmp/sci1_chain.sh
export GEOM="DIM=1024 N_LAYERS=24 N_HEADS=8 HEAD_DIM=128 MLP_HIDDEN=3704 PAM_LAYERS=3,7,11,15,19,23 SHARE_READS=1"
export SEQ=4096 BATCH=16 GRAD_ACCUM=2 LR=8e-4 KEEP_EVERY=15259
export STAGEA_TOKENS=500039680 MAIN_TOKENS=9999998976 DECAY_TOKENS=1000013824
export SOURCES_A=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,pysci
export WEIGHTS_A=0.2335,0.24789,0.05575,0.08468,0.10892,0.12515,0.02954,0.02484,0.03591,0.05383
export SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,tools,think,pysci,b2s,recall4k,proto4k
export WEIGHTS=0.21225,0.22533,0.05068,0.07697,0.09901,0.11376,0.02685,0.02258,0.03264,0.03433,0.02362,0.04893,0.02345,0.00426,0.00535
