#!/usr/bin/env bash
# A/B driver: complex-384 arm, then real-768 arm, sequential (one 4090).
# Each arm logs to its own file; a combined driver log records arm boundaries
# with epoch timestamps for wall-time measurement.
set -uo pipefail
cd /home/gowrav/Development/qllm2
mkdir -p logs

echo "=== A/B driver start $(date -u +%Y-%m-%dT%H:%M:%SZ) epoch=$(date +%s) ==="

echo "--- ARM 1: complex-384 (baseline) start epoch=$(date +%s) ---"
bash v13_sempty/tmp_ab_complex.sh 2>&1 | tee -a logs/ab_complex.log
C_RC=$?
echo "--- ARM 1: complex-384 end rc=$C_RC epoch=$(date +%s) ---"

echo "--- ARM 2: real-768 (baseline_real) start epoch=$(date +%s) ---"
bash v13_sempty/tmp_ab_real.sh 2>&1 | tee -a logs/ab_real.log
R_RC=$?
echo "--- ARM 2: real-768 end rc=$R_RC epoch=$(date +%s) ---"

echo "=== A/B driver end epoch=$(date +%s) (complex rc=$C_RC, real rc=$R_RC) ==="
exit 0
