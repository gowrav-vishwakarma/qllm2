#!/usr/bin/env bash
# Serialized V11 experiment queue for a single GPU.
#
# Waits until the GPU is (mostly) free -- e.g. the Phase 0 baseline finished --
# then runs the architecture experiments back-to-back: E1 -> E3 -> E2.
# (E2/delta is last: it is the slow path until Flash-PAM lands.)
#
# Run inside tmux so it survives disconnects:
#   tmux new-session -d -s v11_queue "./scripts/run_v11_queue.sh 2>&1 | tee /tmp/v11_queue.out"
#
# Override the queue or the free-memory threshold via env:
#   V11_QUEUE="v11_e1_perchannel v11_e3_multistate" V11_FREE_MIB=80000 ./scripts/run_v11_queue.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v11/train.py ]] || cd ..

QUEUE="${V11_QUEUE:-v11_e1_perchannel v11_e3_multistate v11_e2_delta}"
FREE_MIB="${V11_FREE_MIB:-70000}"     # require this much free VRAM before starting next

gpu_free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' '
}

wait_for_gpu() {
    echo "[queue] waiting for >= ${FREE_MIB} MiB free VRAM..."
    while true; do
        free=$(gpu_free_mib)
        if [[ "${free:-0}" -ge "$FREE_MIB" ]]; then
            echo "[queue] GPU free=${free} MiB -- proceeding."
            return 0
        fi
        sleep 120
    done
}

for preset in $QUEUE; do
    wait_for_gpu
    echo "[queue] ===== launching $preset at $(date) ====="
    ./scripts/run_v11_exp.sh "$preset" || echo "[queue] $preset exited non-zero (continuing)"
    echo "[queue] ===== finished $preset at $(date) ====="
    sleep 30
done
echo "[queue] all experiments complete."
