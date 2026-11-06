# SCI-1 long run, segment 2 (EXPERIMENTS_SEMPY "far block", pre-registered 2026-11-07; Lead GO for option (b) + proto_far):
# from the long run's M11 (sci1_long_3c2bf1f step_167849, full) on the cont3 mix + the FAR BLOCK ramped in (R2: one ramp, >= 2B):
# b2s_far 0 -> 1.2 % and proto_far 0 -> 0.8 % of tokens over 2B, taken from dclm / fineweb in proportion; b2s / proto4k / recall4k
# kept (R1). WARMUP 0 (optimizer state kept; s8reg B: a 500-step re-warm cost s8 ~0.10). RAMP_FROM_TOKENS = the checkpoint's own
# cumulative token counters, set by the switch orchestrator at launch. Source AFTER sci1_env.sh (NOT sci1_cont3_env.sh).
export CONT_TAG=sci1_long2
export CONT_FROM=checkpoints_v13_sempty/sci1_long_3c2bf1f/step_167849.pt
export CONT_TOKENS=99999940608 KEEP_LAST=2 WARMUP_C=0
export SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,tools,pysci,b2s,recall4k,proto4k,b2s_far,proto_far
export WEIGHTS=0.23083,0.24033,0.05043,0.07254,0.10145,0.11762,0.02232,0.02244,0.03152,0.03501,0.04631,0.01696,0.00377,0.00846,0.00000,0.00000
export RAMP_WEIGHTS=0.22103,0.23013,0.05043,0.07254,0.10145,0.11762,0.02232,0.02244,0.03152,0.03501,0.04631,0.01696,0.00377,0.00846,0.01200,0.00800
export RAMP_TOKENS=2000000000
