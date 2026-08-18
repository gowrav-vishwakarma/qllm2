#!/usr/bin/env bash
# Tiny TTS smoke: prove the t2s loop (Mimi encode + PAM next-token) before a
# focused run. English-only, ~256 utts, 4 epochs, duplex_100m from scratch.
#
#   tmux new-session -d -s tts_smoke './scripts/run_v11_duplex_tts_smoke.sh'
#   tail -f checkpoints_v11_duplex_100m_tts_pam_t2s_smoke/last_log_dir.txt

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

export PRESET="${PRESET:-duplex_100m}"
export BACKBONE="${BACKBONE:-pam}"
export TASK=t2s
export LANGUAGES="${LANGUAGES:-none}"
export N_PER_LANG="${N_PER_LANG:-0}"
export N_ENGLISH="${N_ENGLISH:-256}"
export EPOCHS="${EPOCHS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export LR="${LR:-3e-4}"
export INIT_FROM=""
export CKPT_DIR="${CKPT_DIR:-checkpoints_v11_${PRESET}_tts_${BACKBONE}_${TASK}_smoke}"

exec ./scripts/run_v11_duplex_tts.sh --warmup_steps 20 --max_codec_frames 120
