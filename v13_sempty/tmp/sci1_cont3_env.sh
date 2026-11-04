# SCI-1 third continuation (EXPERIMENTS_SEMPY "SCI-1 cont2 milestone 6" -> cont3, Lead decision 14:30 IST 2026-11-04): source AFTER
# sci1_env.sh. From cont2 milestone 7 (global 106,813): protoc DROPPED (the one-cause test: is protoc the s8 suppressor?), b2s ~1.4 %,
# proto4k 0.8 %, recall4k 0.34 %; weights from calib6's per-source factors, checked by the measured row calib7 (31.5M tokens).
export CONT_TAG=sci1_cont3
export CONT_FROM=checkpoints_v13_sempty/sci1_cont2_e61cd92/step_106813.pt
export CONT_TOKENS=3499950080        # 53,405 steps x 65,536: global 106,813 -> 160,218, the original stage-B end
export SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,tools,pysci,b2s,recall4k,proto4k
export WEIGHTS=0.23083,0.24033,0.05043,0.07254,0.10145,0.11762,0.02232,0.02244,0.03152,0.03501,0.04631,0.01696,0.00377,0.00846
