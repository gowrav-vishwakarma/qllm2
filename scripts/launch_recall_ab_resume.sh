#!/usr/bin/env bash
# Resume remaining recall A/B arms after control completed (gate AMP fix applied).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_ab_sweep
LOG="logs/v11/recall_ab_sweep/sweep_resume_remaining.log"
echo "$LOG" > logs/v11/recall_ab_sweep/LATEST_LOG.txt

export ARMS="gate floor recall combo"
export TOKEN_BUDGET=300000000
export BATCH_SIZE=16
export WEB_SOURCES=fineweb
export WEB_WEIGHT=96
export FINEWEB_LOCAL_DIR=data/fineweb-edu/sample-10BT
export HF_HUB_DISABLE_XET=1
export RESUME=checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt
export NO_CURSOR=1

# Clear failed/partial gate dir so we start clean
rm -rf checkpoints_v11_recall_ab/gate

nohup ./scripts/run_v11_recall_ab.sh >>"$LOG" 2>&1 &
echo $! > logs/v11/recall_ab_sweep/LATEST_PID.txt
echo "launched pid=$(cat logs/v11/recall_ab_sweep/LATEST_PID.txt) log=$LOG arms=$ARMS"
