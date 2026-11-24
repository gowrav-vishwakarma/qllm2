# sft1 (EXPERIMENTS_SEMPY "sft1"): chat SFT of SCI-1 386M from final_step_749500 (49.119B). Source AFTER sci1_env.sh (geometry,
# BATCH 16 (x GRAD_ACCUM 2 micro-batches) x 4096 = 65,536 tokens / step as the base) and AFTER sci1_long3r_env.sh (the replay mix = 3c / 3r's
# token shares incl. the far rule docs, CYCLE_SOURCES, decontam v2). Launch ONLY on the user's word (via Lead).
export CONT_TAG=sci1_sft1 SFT_SHARE=0.8
export LR=2e-4 WARMUP_C=200                      # 200-step warm-up to 2e-4, constant (wsd_stable) for the main segment
export CONT_TOKENS=200015872 DECAY_TOKENS=50003968   # 3,052 + 763 steps of 65,536: 0.20B constant + 0.05B linear decay to 0
export KEEP_EVERY=763 KEEP_LAST=6                # saves every 0.05B for the guard / chat reads
