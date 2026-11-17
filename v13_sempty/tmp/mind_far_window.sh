#!/bin/bash
# Library's far window (C2/C3/M9/M10 scope, m43far, m44far; gated on ~/.qllm_far_window): opens after the wio mamba2 arm's readouts,
# closes on "=== FAR-DONE" (or 06:00 IST). The yield flag is dropped only if no milestone CHECK and no 3f window holds the card.
W=/home/gowrav/Development/qllm_v14norm/logs/v14_fartold_watch.log
until [ -f checkpoints_v13_sempty/wio_mamba2_decay_227f3eb/.readouts ]; do sleep 30; done
touch ~/.qllm_check_wants_gpu; echo "flag up $(date +%T)"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 4000 ]; do sleep 15; done
touch ~/.qllm_far_window; echo "far window open $(date +%T)"
until grep -q '=== FAR-DONE' $W || { h=$(date +%H%M); [ $h -ge 0600 ] && [ $h -lt 1200 ]; }; do [ -f ~/.qllm_check_wants_gpu ] || touch ~/.qllm_check_wants_gpu; sleep 15; done
rm -f ~/.qllm_far_window
if pgrep -f '^bash v13_sempty/tmp/sci1_check_local.sh' >/dev/null || [ -f ~/.qllm_lib_window ]; then echo "far window closed; another holder keeps the flag $(date +%T)"
else rm -f ~/.qllm_check_wants_gpu; echo "far window closed, card back to wio $(date +%T)"; fi
