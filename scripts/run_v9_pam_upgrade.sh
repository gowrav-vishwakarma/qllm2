#!/usr/bin/env bash
# V9 PAM upgrade experiments.
#
# Variants:
#   baseline  - clean V9 wrapper of medium_h16_flat
#   gate      - PAM output gate only (~105M)
#   gate_100m - PAM output gate only, dim=372 (~100.5M)
#   gate_revassoc_100m - PAM output gate + reverse association, dim=372
#   gate_qknorm_100m   - PAM output gate + QK normalization, dim=372
#   gate_conv4_100m    - PAM output gate + causal depthwise short conv, dim=372
#   conv      - causal depthwise short conv only
#   gate_conv - output gate + short conv
#   compete_revassoc_100m - V6 PAM + reverse_assoc + cross-head soft competition (zero params)
#
# Usage:
#   bash ./scripts/run_v9_pam_upgrade.sh --variant gate_100m
#   bash ./scripts/run_v9_pam_upgrade.sh --variant gate
#   bash ./scripts/run_v9_pam_upgrade.sh --variant conv --epochs 3
#   bash ./scripts/run_v9_pam_upgrade.sh --variant gate_conv --batch_size 2

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v9/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VARIANT="gate"
EPOCHS=10
SEQ_LEN=2048
DATASET="wikitext103"
BATCH_SIZE=3
RESUME=0
EXTRA_ARGS=""
VARIANT_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)    VARIANT="$2";    shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --seq_len)    SEQ_LEN="$2";    shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --dataset)    DATASET="$2";    shift 2 ;;
        --resume)     RESUME=1;        shift ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

case "$VARIANT" in
    baseline)
        PRESET="medium_h16_flat"
        DESC="V9 clean baseline: medium_h16_flat, reverse-assoc off"
        VARIANT_ARGS="--no_reverse_assoc"
        ;;
    gate)
        PRESET="medium_h16_gate"
        DESC="V9 Exp A: PAM output gate only, reverse-assoc off"
        VARIANT_ARGS="--no_reverse_assoc"
        ;;
    gate_100m)
        PRESET="medium_h16_gate_100m"
        DESC="V9 Exp A100: PAM output gate, parameter-matched 100M, reverse-assoc off"
        VARIANT_ARGS="--no_reverse_assoc"
        ;;
    gate_revassoc_100m)
        PRESET="medium_h16_gate_revassoc_100m"
        DESC="V9 Exp R: gate + reverse_assoc, ~100M (test if gate detoxifies reverse pass)"
        VARIANT_ARGS=""
        ;;
    gate_qknorm_100m)
        PRESET="medium_h16_gate_qknorm_100m"
        DESC="V9 Exp N: gate + qk_norm, ~100M (gate=magnitude, qk_norm=angle stability)"
        VARIANT_ARGS="--no_reverse_assoc"
        ;;
    gate_conv4_100m)
        PRESET="medium_h16_gate_conv4_100m"
        DESC="V9 Exp K: gate + short_conv=4, ~100M (orthogonal local pattern)"
        VARIANT_ARGS="--no_reverse_assoc"
        ;;
    conv)
        PRESET="medium_h16_conv4"
        DESC="V9 Exp B: causal short conv only, reverse-assoc off"
        VARIANT_ARGS="--no_reverse_assoc"
        ;;
    gate_conv)
        PRESET="medium_h16_gate_conv4"
        DESC="V9 Exp C: PAM output gate + causal short conv, reverse-assoc off"
        VARIANT_ARGS="--no_reverse_assoc"
        ;;
    compete_revassoc_100m)
        PRESET="medium_h16_compete_revassoc_100m"
        DESC="V9 Exp Comp: V6 PAM + reverse_assoc + cross-head soft competition (zero params, ~100M)"
        VARIANT_ARGS=""
        ;;
    *)
        echo "Unknown --variant '$VARIANT'. Expected: baseline, gate, gate_100m, gate_revassoc_100m, gate_qknorm_100m, gate_conv4_100m, compete_revassoc_100m, conv, gate_conv" >&2
        exit 2
        ;;
esac

GEN_PROMPT="In 1923 , the University of"
CKPT_DIR="checkpoints_v9_${VARIANT}"
LOG_DIR_SIDECAR="${CKPT_DIR}/last_log_dir.txt"
ARGS="--preset $PRESET --dataset $DATASET --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS --activation swish --max_samples 9999999 --compile --compile_mode default --amp_dtype auto --num_workers 4 --gen_every 5000 --no_grad_ckpt $VARIANT_ARGS"

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
    LOG_DIR=$(make_log_dir "v9" "pam_${VARIANT}_${DATASET}")
fi

mkdir -p "$CKPT_DIR"
printf '%s\n' "$LOG_DIR" > "$LOG_DIR_SIDECAR"

if [[ $RESUME -eq 1 && -f "$CKPT_DIR/best_model.pt" ]]; then
    RESUME_ARG="--resume $CKPT_DIR/best_model.pt"
    echo "[resume] Resuming from $CKPT_DIR/best_model.pt"
fi

RUN_DESC="$DESC: dim=384 L=16 expand=3, PAM(H=6, d=64, flat dt=-4.0, RoPE, fused-QKV, GSP), activation=swish, LR=1e-4, warmup=1000"
RUN_ARGS_LINE="$ARGS --gen_prompt '$GEN_PROMPT' $RESUME_ARG $EXTRA_ARGS"

echo ""
echo "============================================================"
echo "  $DESC"
echo "  Preset: $PRESET (~100M params plus variant overhead)"
echo "  Architecture: [CGU(expand=3, swish) -> V9 PAM] x16 + GSP + RoPE"
echo "  seq_len: $SEQ_LEN  batch_size: $BATCH_SIZE  epochs: $EPOCHS"
echo "  Dataset: $DATASET"
echo "  Log: $LOG_DIR"
echo "  Checkpoint: $CKPT_DIR"
echo "  Target: Beat V7 Exp7a val PPL 29.73 and move toward Transformer 27.08"
echo "============================================================"
echo ""

if [[ $REUSED_LOG_DIR -eq 1 ]]; then
    append_run_info_resume "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"
else
    write_run_info "$LOG_DIR" "$RUN_DESC" "$RUN_ARGS_LINE"
fi

start_time=$(date +%s)

eval "$PYTHON_BIN -m v9.train" \
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
echo "  V9 PAM upgrade ($VARIANT) complete!"
echo "  Time: ${mins}m ${secs}s"
echo "  Results in: $LOG_DIR"
echo ""
echo "  Baseline comparisons:"
echo "    V7 Exp7a (ModSwish, B=3): Val PPL 29.73"
echo "    Transformer (B=3):         Val PPL 27.08"
echo "============================================================"
