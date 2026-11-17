# SCI-1 long run, segment 3e = HARM REMEDY (EXPERIMENTS_SEMPY "3d ADDENDUM: HARM readout", Lead 2026-11-17): 3d's mix with proto_far
# (v1, 100 % protocol docs) REPLACED by proto_far2 (Library e670df5a: 65 % own protocol / 25 % no protocol / 10 % other-quantity,
# no-protocol + other answer the bare value) at the same 1.6 %. proto_far2 starts at doc 0 on its own cursor; proto_far's cursor stays in
# the save, unused. Everything else as 3d (chat 19.9 %, b2s_far 1.2 %, caps, decontam v2).
# Source AFTER sci1_env.sh. CONT_FROM is set by sci1_seg_switch.sh. Launch ONLY on Lead's GO, after the three checks (packer identity,
# Library check(), realized share in the trainer's [mix] line).
export CONT_TAG=sci1_long3e CONT_TOKENS=99999940608 KEEP_LAST=2 WARMUP_C=0 DECONTAM_INDEX_C=.cache/decontam/v2.npy
export SOURCES=dclm,fineweb100,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt2,tools,pysci,b2s,recall4k,proto4k,b2s_far,proto_far2,chatsr,tuluif
export WEIGHTS=0.14597,0.15215,0.05043,0.07254,0.10145,0.11762,0.02232,0.12000,0.06000,0.03501,0.04631,0.01696,0.00377,0.00846,0.01200,0.01600,0.01500,0.00400
export CYCLE_SOURCES=smoltalk2_mid:4,chatmt2:3,chatsr:4,tuluif:4
unset RAMP_WEIGHTS RAMP_FROM_TOKENS RAMP_TOKENS
