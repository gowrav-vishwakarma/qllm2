#!/usr/bin/env bash
# Uninterrupted recall A/B on this box: local FineWeb shards (no HF auth),
# batch 16 (bs=64 OOM'd under torch.compile on the RTX PRO 6000).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/v11/recall_ab_sweep
LOG="logs/v11/recall_ab_sweep/sweep_localfw_bs16.log"
PIDFILE="logs/v11/recall_ab_sweep/LATEST_PID.txt"
echo "$LOG" > logs/v11/recall_ab_sweep/LATEST_LOG.txt

export ARMS="control gate floor recall combo"
export TOKEN_BUDGET=300000000
export BATCH_SIZE=16
export WEB_SOURCES=fineweb
export WEB_WEIGHT=96
export FINEWEB_LOCAL_DIR=data/fineweb-edu/sample-10BT
export HF_HUB_DISABLE_XET=1
export RESUME=checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt
export NO_CURSOR=1

nohup ./scripts/run_v11_recall_ab.sh >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "launched pid=$(cat "$PIDFILE") log=$LOG"
