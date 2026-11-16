# SCI-1 long run, segment 3c (EXPERIMENTS_SEMPY "segment 3c: chat sources cycle", 2026-11-16; user: "base model WITH chat", Lead
# PRIORITY 1): segment 3's mix at the chat ramp's TARGET shares (the ramp ended at ~30B), started from segment 3's latest save, with the
# four chat sources re-opening from their start when dry (CYCLE_SOURCES, a31ed81), up to R7's 4-epoch cap: smoltalk2_mid 4 passes,
# chatmt2 3 (chatmt's first epoch was its own source), chatsr 4, tuluif 4. Everything else unchanged (far block, rule docs, decontam v2).
# Source AFTER sci1_env.sh. CONT_FROM is set by sci1_seg3c_switch.sh.
export CONT_TAG=sci1_long3c CONT_TOKENS=99999940608 KEEP_LAST=2 WARMUP_C=0 DECONTAM_INDEX_C=.cache/decontam/v2.npy
export SOURCES=dclm,fineweb100,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt2,tools,pysci,b2s,recall4k,proto4k,b2s_far,proto_far,chatsr,tuluif
export WEIGHTS=0.14997,0.15615,0.05043,0.07254,0.10145,0.11762,0.02232,0.12000,0.06000,0.03501,0.04631,0.01696,0.00377,0.00846,0.01200,0.00800,0.01500,0.00400
export CYCLE_SOURCES=smoltalk2_mid:4,chatmt2:3,chatsr:4,tuluif:4
unset RAMP_WEIGHTS RAMP_FROM_TOKENS RAMP_TOKENS
