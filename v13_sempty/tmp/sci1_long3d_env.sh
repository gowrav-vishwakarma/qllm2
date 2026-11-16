# SCI-1 long run, segment 3d (EXPERIMENTS_SEMPY "segment 3d: proto_far 0.8 % -> 1.6 %", pre-registered 19:28 IST 2026-11-16, Lead GO
# with the unit4 condition): segment 3c's mix with proto_far doubled, the extra 0.8 % taken equally from dclm and fineweb100. Chat 19.9 %,
# b2s_far 1.2 %, CYCLE_SOURCES caps, decontam v2 unchanged. Started from 3c's 41B save (step_625619) by sci1_seg3d_switch.sh.
# Source AFTER sci1_env.sh.
export CONT_TAG=sci1_long3d CONT_TOKENS=99999940608 KEEP_LAST=2 WARMUP_C=0 DECONTAM_INDEX_C=.cache/decontam/v2.npy
export SOURCES=dclm,fineweb100,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt2,tools,pysci,b2s,recall4k,proto4k,b2s_far,proto_far,chatsr,tuluif
export WEIGHTS=0.14597,0.15215,0.05043,0.07254,0.10145,0.11762,0.02232,0.12000,0.06000,0.03501,0.04631,0.01696,0.00377,0.00846,0.01200,0.01600,0.01500,0.00400
export CYCLE_SOURCES=smoltalk2_mid:4,chatmt2:3,chatsr:4,tuluif:4
unset RAMP_WEIGHTS RAMP_FROM_TOKENS RAMP_TOKENS
