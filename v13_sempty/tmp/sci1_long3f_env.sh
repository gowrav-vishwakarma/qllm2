# SCI-1 long run, segment 3f = SCOPE sources (EXPERIMENTS_SEMPY "SCOPE SWITCH RULE", Argon / Lead 2026-11-17): 3d's mix with
# proto_far -> proto_far3 (55 % own / 25 % none / 20 % other-quantity, far) and proto4k -> proto4k2 (proto4k + 20 % other-quantity
# docs at short gaps) at the SAME shares; both hold out the unit direction "asked volume, mass protocol" (the scope probe's S cell).
# New sources start at doc 0 on their own cursors; proto_far / proto4k cursors stay in the save, unused. Everything else as 3d.
# Source AFTER sci1_env.sh. CONT_FROM is set by sci1_seg_switch.sh. Launch ONLY on Lead's GO, after the three checks.
export CONT_TAG=sci1_long3f CONT_TOKENS=99999940608 KEEP_LAST=2 WARMUP_C=0 DECONTAM_INDEX_C=.cache/decontam/v2.npy
export SOURCES=dclm,fineweb100,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt2,tools,pysci,b2s,recall4k,proto4k2,b2s_far,proto_far3,chatsr,tuluif
export WEIGHTS=0.14597,0.15215,0.05043,0.07254,0.10145,0.11762,0.02232,0.12000,0.06000,0.03501,0.04631,0.01696,0.00377,0.00846,0.01200,0.01600,0.01500,0.00400
export CYCLE_SOURCES=smoltalk2_mid:4,chatmt2:3,chatsr:4,tuluif:4
unset RAMP_WEIGHTS RAMP_FROM_TOKENS RAMP_TOKENS
