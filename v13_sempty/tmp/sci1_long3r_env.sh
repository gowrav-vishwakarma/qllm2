# SCI-1 long run, segment 3r = HARM FALLBACK (EXPERIMENTS_SEMPY "3d ADDENDUM: HARM readout", Lead 2026-11-17): back to segment 3c's
# weights (proto_far 0.8 %, the extra 0.8 % returned to dclm / fineweb100) from 3d's latest save, if noprot gate_win >= 0.30 at two
# consecutive saves and proto_far v2 has not passed its checks. Same sources, caps, decontam v2 as 3c/3d; all cursors carry over.
# Source AFTER sci1_env.sh. CONT_FROM is set by sci1_seg_switch.sh. Launch ONLY on Lead's GO.
export CONT_TAG=sci1_long3r CONT_TOKENS=99999940608 KEEP_LAST=2 WARMUP_C=0 DECONTAM_INDEX_C=.cache/decontam/v2.npy
export SOURCES=dclm,fineweb100,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt2,tools,pysci,b2s,recall4k,proto4k,b2s_far,proto_far,chatsr,tuluif
export WEIGHTS=0.14997,0.15615,0.05043,0.07254,0.10145,0.11762,0.02232,0.12000,0.06000,0.03501,0.04631,0.01696,0.00377,0.00846,0.01200,0.00800,0.01500,0.00400
export CYCLE_SOURCES=smoltalk2_mid:4,chatmt2:3,chatsr:4,tuluif:4
unset RAMP_WEIGHTS RAMP_FROM_TOKENS RAMP_TOKENS
