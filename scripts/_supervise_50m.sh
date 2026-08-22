#!/usr/bin/env bash
cd "./.."
TRAIN_PID=1754550
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
if [[ -f checkpoints_v13/e2b_50m/final_model.pt || -f checkpoints_v13/e2b_50m/best_model.pt ]]; then
  bash scripts/run_v13_e2b_50m_rest.sh
else
  echo "V13 50M train failed - removing partial logs"
  rm -f logs/v13/e2b_50m_resume2.log
fi
