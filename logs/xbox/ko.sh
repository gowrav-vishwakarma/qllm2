cd ~/Development/qllm_twins; H=$(git rev-parse --short HEAD); CK=checkpoints_v13_sempty/nt_split_decay_10a4c1db/latest.pt
for ko in none clock rope; do
  if [ $ko = none ]; then unset V13_CHRONO_KNOCKOUT; else export V13_CHRONO_KNOCKOUT=$ko; fi
  O=logs/wio/v13_sempty_split_ko_${ko}_${H}
  .venv/bin/python scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $CK --output ${O}_spread_s7000_t400.json --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|Error" | cut -c1-400
  .venv/bin/python scripts/run_memory_behavioral.py --model-type v13_sempty --checkpoint $CK --preset baseline_real_pm --context-lengths 16384 --positions 0,0.5,1 --association-counts 1,4,8 --trials 40 --max-context 16384 --output ${O}_t40_16k.json 2>&1 | grep -v scored | tail -1 | cut -c1-200
  echo "=== ko $ko done $(date -u +%FT%TZ)"
done; echo "=== KO ALL DONE"
