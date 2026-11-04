# SCI-1 second continuation (EXPERIMENTS_SEMPY "SCI-1 cont1 milestone 5" -> cont2, Lead decision B 06:00 IST 2026-11-04): source
# AFTER sci1_env.sh. From cont1 milestone 5 (global 76,295): b2s back to 1.2 %, proto4k back to 0.4 %, protoc 0.15 %, recall4k
# 0.34 % (targets; 4.17 % synthetic loss weight with the measured per-token weights); weights from calib5's per-source factors.
export CONT_TAG=sci1_cont2
export CONT_FROM=checkpoints_v13_sempty/sci1_cont1_59d098d/step_076295.pt
export CONT_TOKENS=5499977728        # 83,923 steps x 65,536: global 76,295 -> 160,218, the original stage-B end
export SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,tools,pysci,b2s,recall4k,proto4k,protoc
export WEIGHTS=0.23324,0.24116,0.04942,0.07532,0.10153,0.11154,0.02428,0.02231,0.03302,0.03252,0.04883,0.01715,0.00379,0.00478,0.00112
