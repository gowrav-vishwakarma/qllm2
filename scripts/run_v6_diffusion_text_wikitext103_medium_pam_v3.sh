#!/usr/bin/env bash
# V6 diffusion_text on WikiText-103 — medium-pam-v3 preset (~100M class).
#
# Same backbone preset class as medium-pam-v3 AR; mode=diffusion_text (latent DDPM).
# Discord: export DISCORD_HOOK='https://...' OR set DISCORD_HOOK in repo-root .env
#
# NOTE on `div=n/a`: medium-pam-v3 is a single-bank preset, so the dual-bank
# diversity loss is not computed. The trainer prints `div=n/a` (not zero) to
# make this explicit. See EXPERIMENTS_V6_DIFFUSION_PART2.md "Phase 0".
#
# Hardware profiles via --hw:
#   --hw pro6000  (default) batch=18 compile=max-autotune workers=12   ~Pro 6000 96 GB
#   --hw 4090               batch=3  compile=reduce-overhead workers=4 ~RTX 4090 24 GB
#
# Usage:
#   ./scripts/run_v6_diffusion_text_wikitext103_medium_pam_v3.sh
#   ./scripts/run_v6_diffusion_text_wikitext103_medium_pam_v3.sh --hw 4090
#   ./scripts/run_v6_diffusion_text_wikitext103_medium_pam_v3.sh --epochs 3
#   ./scripts/run_v6_diffusion_text_wikitext103_medium_pam_v3.sh --batch_size 8
#   ./scripts/run_v6_diffusion_text_wikitext103_medium_pam_v3.sh --resume

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS=10
SEQ_LEN=2048
DATASET="wikitext103"
SIZE="medium-pam-v3"
HW="pro6000"
BATCH_SIZE=""
COMPILE_MODE=""
NUM_WORKERS=""
RESUME=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)       EPOCHS="$2";       shift 2 ;;
        --seq_len)      SEQ_LEN="$2";      shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
        --compile_mode) COMPILE_MODE="$2"; shift 2 ;;
        --num_workers)  NUM_WORKERS="$2";  shift 2 ;;
        --hw)           HW="$2";           shift 2 ;;
        --resume)       RESUME=1;          shift ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

case "$HW" in
    pro6000)
        : "${BATCH_SIZE:=18}"
        : "${COMPILE_MODE:=max-autotune}"
        : "${NUM_WORKERS:=12}"
        ;;
    4090)
        : "${BATCH_SIZE:=3}"
        : "${COMPILE_MODE:=reduce-overhead}"
        : "${NUM_WORKERS:=4}"
        ;;
    *)
        echo "[hw] Unknown --hw '$HW' (expected: pro6000|4090)" >&2
        exit 2
        ;;
esac

GEN_PROMPT="In 1923 , the University of"
CKPT_DIR="checkpoints_v6_wikitext103_diffusion_medium_pam_v3"
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"

ARGS="--dataset $DATASET --size $SIZE --mode diffusion_text --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --max_samples 9999999 --diffusion_steps 1000 --noise_schedule cosine --prediction_target x0 --sampling_method ddpm --no_working_memory --no_internal_memory --init_seed 42 --gen_every 5000 --compile --compile_mode $COMPILE_MODE --amp_dtype auto --num_workers $NUM_WORKERS --log_diff_diagnostics"

RESUME_ARG=""
REUSED_LOG_DIR=0
LOG_DIR=""

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" && -f "$LOG_DIR_SIDECAR" ]]; then
    _stored=$(head -n 1 "$LOG_DIR_SIDECAR" | tr -d '\r')
    if [[ -n "$_stored" && -d "$_stored" ]]; then
        LOG_DIR="$_stored"
        REUSED_LOG_DIR=1
        echo "[resume] Reusing log directory from $LOG_DIR_SIDECAR: $LOG_DIR"
    elif [[ -n "$_stored" ]]; then
        echo "[resume] Warning: stored log dir not found on disk: $_stored (will create a new log dir)" >&2
    fi
fi

if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR=$(make_log_dir "v6" "wikitext103_diffusion_text_medium_pam_v3")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="V6 diffusion_text medium-pam-v3 on WikiText-103 (no WM/IM); DDPM 1000 cosine x0; hw=$HW compile=$COMPILE_MODE workers=$NUM_WORKERS"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  V6 diffusion_text — medium-pam-v3 / WikiText-103"
echo "  hw=$HW  compile=$COMPILE_MODE  workers=$NUM_WORKERS"
echo "  mode=diffusion_text  diffusion_steps=1000  cosine  x0  DDPM"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  (single-bank preset -> div=n/a is expected; pred/target norms via --log_diff_diagnostics)"
echo "============================================================"
echo ""

if [[ $REUSED_LOG_DIR -eq 1 ]]; then
    append_run_info_resume "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"
else
    write_run_info "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"
fi

start_time=$(date +%s)

eval "$PYTHON_BIN -m v6.train" \
    $ARGS \
    --gen_prompt "'$GEN_PROMPT'" \
    --log_dir "$LOG_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    $RESUME_ARG \
    $EXTRA_ARGS

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
mins=$(( elapsed / 60 ))
secs=$(( elapsed % 60 ))

echo ""
echo "============================================================"
echo "  V6 diffusion_text (medium-pam-v3) run complete"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo "============================================================"
echo ""
