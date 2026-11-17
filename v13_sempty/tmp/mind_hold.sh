#!/bin/bash
# Hold the 4090 for a Library window (wio_chain yields while ~/.qllm_check_wants_gpu exists): wait for a start condition, then keep the
# flag up (re-touch if a waiter removes it) until ~/.qllm_window_close exists or the deadline passes.
#   usage: mind_hold.sh "<start shell condition>" <deadline HH:MM>
START=${1:?}; DL=${2:?}
until eval "$START"; do sleep 20; done; echo "hold starts $(date +%T)"; rm -f ~/.qllm_window_close
until [ -f ~/.qllm_window_close ] || [ "$(date +%H%M)" -ge "${DL/:/}" ]; do [ -f ~/.qllm_check_wants_gpu ] || { touch ~/.qllm_check_wants_gpu; echo "flag up $(date +%T)"; }; sleep 10; done
pgrep -f '^bash v13_sempty/tmp/sci1_check_local.sh' >/dev/null && { echo "a CHECK runs: flag left to its waiter $(date +%T)"; exit 0; }
rm -f ~/.qllm_check_wants_gpu ~/.qllm_window_close; echo "hold ends $(date +%T)"
