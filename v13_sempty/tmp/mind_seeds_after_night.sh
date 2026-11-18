#!/bin/bash
# 4090: when the night chain (gdn_nofar -> split) ends, start the seed-44 wio far-mix arms (pam, nochrono, gdn) with twins_chain.sh.
cd "$(dirname "$0")/../.." || exit 1
while tmux has-session -t =mindnight 2>/dev/null; do sleep 60; done
echo "night chain ended $(date)"
H=$(git rev-parse --short HEAD)
export BOX=4090 SCREEN_HASH=$H
export CHAIN_ARMS='pam_s44|CHRONO=1 SEED=44 $PAMSPEC;nochrono_s44|CHRONO=0 SEED=44 $PAMSPEC;gdn_s44|ARCH=gdn PYTHON=$HOME/.venvs/qllm_gdn/bin/python SEED=44 $BASELINE'
bash v13_sempty/tmp/twins_chain.sh 2>&1 | tee -a logs/v13_sempty_twins_seeds_chain_${H}_$(date +%Y%m%d_%H%M).log
