# SCI-1 long run, segment 3 (EXPERIMENTS_SEMPY "segment 3", pre-registered 2026-11-12; Lead's decisions on the R5 audit 26630a1):
# (a) fineweb -> fineweb100 (fineweb-edu sample-100BT streamed from HF, own cursor; the local sample-10BT ran dry at ~23.1B) and
# chatmt -> chatmt2 (epoch 2, own cursor; ran dry at ~21.4B), both at their old shares; (b) the CHAT RAMP (R2, the one ramp in flight,
# 2B tokens): chat 5.4 % -> 19.9 % (smoltalk2_mid 2.2 -> 12 %, chatmt2 3.2 -> 6 %, chatsr 0 -> 1.5 %, tuluif 0 -> 0.4 %; R7-sized for
# <= 4 epochs over 10B), taken from dclm / fineweb100 in proportion; far block and rule docs unchanged (R1); smol-constraints OUT.
# Stream decontam index v2 (v1 + IFEval, sha a4b66d1b). CONT_FROM is set by the switch orchestrator (28B or 344k: Lead's rollback rule).
# Source AFTER sci1_env.sh.
export CONT_TAG=sci1_long3 CONT_TOKENS=99999940608 KEEP_LAST=2 WARMUP_C=0 DECONTAM_INDEX_C=.cache/decontam/v2.npy
export SOURCES=dclm,fineweb100,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt2,tools,pysci,b2s,recall4k,proto4k,b2s_far,proto_far,chatsr,tuluif
export WEIGHTS=0.22103,0.23013,0.05043,0.07254,0.10145,0.11762,0.02232,0.02244,0.03152,0.03501,0.04631,0.01696,0.00377,0.00846,0.01200,0.00800,0.00000,0.00000
export RAMP_WEIGHTS=0.14997,0.15615,0.05043,0.07254,0.10145,0.11762,0.02232,0.12000,0.06000,0.03501,0.04631,0.01696,0.00377,0.00846,0.01200,0.00800,0.01500,0.00400
export RAMP_TOKENS=2000000000
