#!/usr/bin/env bash
# v13_sempty streaming-pretrain launcher (Phase 3: scale the DATA, arch fixed).
#
# Default = the 2026-09-04 scale run: real-102M `baseline_real_pm` + CHRONO +
# OUT_GATE (the 22.96 WikiText reference arch) on a live-streamed blend
#   dclm-edu 45 % / fineweb-edu 42 % / smoltalk2 Mid (ChatML text) 10 % /
#   synthetic recall 3 %      (web-only for the first BLEND_WARMUP tokens)
# for TARGET_TOKENS at B=18 T=2048 (36,864 tok/step), dropout 0, lr 2e-4 cosine.
# Primary metric = streaming holdout val (5 % of the corpus); the WikiText-103
# val is logged as a secondary anchor (NOT comparable 1:1 with the 22.96
# WikiText-only runs - different data, single pass).
#
# Usage (RTX Pro 6000, ALWAYS in tmux, env INSIDE the command string):
#   tmux new-session -d -s sempty_mix \
#     "TAG=mix3b_chrono_gate bash v13_sempty/tmp_pretrain_mix.sh"
# Manual resume after the box itself went down (same commit checked out; pass
# CKPT_DIR/RESUME_LOG explicitly if HEAD moved so the run keeps its dir+log):
#   tmux new-session -d -s sempty_mix \
#     "TAG=mix3b_chrono_gate CKPT_DIR=checkpoints_v13_sempty/mix3b_chrono_gate_<hash> \
#      RESUME_LOG=logs/v13_sempty_mix3b_chrono_gate_<hash>_<stamp>.log \
#      bash v13_sempty/tmp_pretrain_mix.sh"
# Requires the `sempyt` .pth in the venv and network access to HF (public
# datasets, no token needed).
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
# Run from a private temp copy so editing this file mid-run cannot corrupt the
# wrapper (bash reads scripts incrementally; see tmp_wikitext_fair.sh).
if [ -z "${_MIX_COPY:-}" ]; then
  _copy=$(mktemp /tmp/tmp_pretrain_mix.XXXXXX.sh) && cp "${BASH_SOURCE[0]}" "$_copy" \
    && REPO_ROOT="$REPO_ROOT" _MIX_COPY=1 exec bash "$_copy" "$@"
fi
# Capture the temp copy's path NOW: a trap body evaluates BASH_SOURCE when it fires, and inside a sourced file that refuses
# (scripts/require_clean_tree.sh: exit 3) it named the guard itself, which the old trap deleted (Lead 2026-11-18). Only /tmp copies.
_self="${BASH_SOURCE[0]}"; trap 'case "$_self" in /tmp/tmp_*.sh) rm -f "$_self";; esac' EXIT

PRESET="${PRESET:-baseline_real_pm}"
TAG="${TAG:-mix3b_chrono_gate}"
SOURCES="${SOURCES:-dclm,fineweb,smoltalk2_mid,recall}"
WEIGHTS="${WEIGHTS:-0.45,0.42,0.10,0.03}"
TARGET_TOKENS="${TARGET_TOKENS:-3000000000}"   # 3.0B -> 81,380 steps at 18x2048
BLEND_WARMUP="${BLEND_WARMUP:-300000000}"      # web-only for the first 300M tok
BATCH="${BATCH:-18}"
SEQ="${SEQ:-2048}"
WARMUP="${WARMUP:-1000}"
LR="${LR:-2e-4}"
DROPOUT="${DROPOUT:-0.0}"
GRAD_CKPT="${GRAD_CKPT:-0}"    # 96 GB box: off (+23 % tok/s); 1 fits the 4090
CHRONO="${CHRONO:-1}"
CHRONO_HEADS="${CHRONO_HEADS:--1}"   # SPLIT key geometry: notebook heads with the clock (-1 = all)
OUT_GATE="${OUT_GATE:-1}"
# Retention ladder (one variable per run, stacked on chrono+gate; compare the
# recall-horizon curve, holdout PPL is the guard):
NSTATES="${NSTATES:-0}"        # A2 states per head (0 = preset default 1)
VAULT="${VAULT:-0}"            # A2b pinned state + protect gate (needs NSTATES>=2)
DELTA="${DELTA:-0}"            # A3 delta erase/write (KEEP 2026-09-05: exact 8-way recall on the micro-bench)
                               # (COUPLE removed 2026-10-28: C1 FAIL, code at 4c688cb)
SHORT_CONV="${SHORT_CONV:-0}"  # G1 short conv on qkv (2026-10-29 screen; needs GEN_EVERY=0)
SEED="${SEED:-42}"            # init + data-mix seed (default 42 = every run so far)
ECHO="${ECHO:-0}"              # E1 echo (succession) memory into the head
NGRAM_ORDERS="${NGRAM_ORDERS:-}"      # v14 N7 in-context n-gram memory orders, e.g. 2,3 (empty = off; builds V14LM)
RESET_AT_EOS="${RESET_AT_EOS:-0}"      # lg8r: state reset at document separators
GRAD_ACCUM="${GRAD_ACCUM:-1}"          # micro-batches per step (exact accumulation; 1 = off)
COND_MEM="${COND_MEM:-0}"              # A4 hashed n-gram conditional memory (tables lr x5, no wd)
ECHO_SURPRISE_TAU="${ECHO_SURPRISE_TAU:-0}" # v14 N12 surprise-selected echo writes, nats (0 = off; needs ECHO=1; builds V14LM)
PAM_LAYERS="${PAM_LAYERS:-}"   # M1 blocks that keep a notebook, e.g. 3,7,11,15 (default all)
MLP_HIDDEN="${MLP_HIDDEN:-0}"  # M1 gated-MLP width (0 = dim*expand), to param-match
SHARE_READS="${SHARE_READS:-0}" # M2 notebook-less blocks read the nearest notebook below; GEN_EVERY=0
N_LAYERS="${N_LAYERS:-0}"      # main-stack blocks (0 = preset 16)
BASE_DT_BIAS="${BASE_DT_BIAS:-}" # override initial decay bias (default -4; bench: -6/-8 no gain)
                               # (LONG_HEADS removed 2026-10-27: R3 FAIL, EXPERIMENTS "R3")
                               # (STORE removed 2026-10-27: R4 FAIL, EXPERIMENTS "R4")
INIT_FROM="${INIT_FROM:-}"     # warm start: weights from this ckpt, stream continues after its cursor
ARCH="${ARCH:-pam}"            # pam | mamba2 | transformer | grove (set CHRONO=0 OUT_GATE=0 DELTA=0)
GROVE_SPEC="${GROVE_SPEC:-}"   # v15 Grove body spec for ARCH=grove (empty = v15.hook.DEFAULT_SPEC, ~102.6M)
MAMBA_N_LAYER="${MAMBA_N_LAYER:-17}"   # 17 x 768 = 102.6M (param-matched to PAM)
MAMBA_D_MODEL="${MAMBA_D_MODEL:-768}"  # 1024 x 51 layers = 388.1M (+0.6 % of SCI-1 PAM)
TPP_GEOM="${TPP_GEOM:-}"               # d_model,n_layer,n_head,d_ff; empty = 101.9M default; 1024,24,16,3168 = 385.75M
                               # (NO_DECAY removed 2026-09-07: FAIL at 100M, EXPERIMENTS "R2")
ANSWER_W="${ANSWER_W:-1}"      # CE weight on answer tokens of synthetic recall/reason docs
                               # (1 = plain LM loss). Micro-bench: recall is learned only
                               # with concentrated answer signal -> use e.g. 100.
GEN_EVERY="${GEN_EVERY:-4000}" # set 0 for vault/n_states>1 (no decode path); delta decodes (2026-09-06)
VAL_EVERY="${VAL_EVERY:-2000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"       # latest.pt (full resume state) every ~7 min
KEEP_EVERY="${KEEP_EVERY:-10000}"      # milestone step_XXXXXX.pt copies (~1.2 GB each)
KEEP_LAST="${KEEP_LAST:-1}"            # rolling: only the newest KEEP_LAST milestones survive
                                       # (latest.pt is overwritten atomically; best_model.pt kept)
MAX_RETRIES="${MAX_RETRIES:-5}"        # auto-resume attempts after a crash
CKPT_DIR="${CKPT_DIR:-}"               # default checkpoints_v13_sempty/<TAG>_<HASH>
RESUME_LOG="${RESUME_LOG:-}"           # append to an existing run log instead of a new one

# Refuse uncommitted code / code drift from a pinned run (PIN_HASH = the
# commit an existing CKPT_DIR/RESUME_LOG is named after).
[ -f scripts/require_clean_tree.sh ] || { echo "launch guard scripts/require_clean_tree.sh missing: REFUSE (restore: git show HEAD:scripts/require_clean_tree.sh > scripts/require_clean_tree.sh)"; exit 1; }   # fail closed (2026-11-16: the guard was found deleted, so source only warned)
source scripts/require_clean_tree.sh "${PIN_HASH:-}"
HASH=$(git rev-parse --short HEAD)
STAMP=$(date +%Y%m%d_%H%M)
CKPT_DIR="${CKPT_DIR:-checkpoints_v13_sempty/${TAG}_${HASH}}"
if [ -n "$RESUME_LOG" ]; then LOG="$RESUME_LOG"; else LOG="logs/v13_sempty_${TAG}_${HASH}_${STAMP}.log"; fi
echo "=== ${TAG} start $(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$HASH log=$LOG ckpt=$CKPT_DIR ===" | tee -a "$LOG"
EXTRA=()
if [ "$GRAD_CKPT" = "1" ]; then EXTRA+=(--gradient_checkpointing); fi
if [ "$CHRONO" = "1" ]; then EXTRA+=(--chrono); fi
if [ "$CHRONO_HEADS" != "-1" ]; then EXTRA+=(--chrono_heads "$CHRONO_HEADS"); fi
if [ "$OUT_GATE" = "1" ]; then EXTRA+=(--out_gate); fi
if [ "$NSTATES" != "0" ]; then EXTRA+=(--n_states "$NSTATES"); fi
if [ "$VAULT" = "1" ]; then EXTRA+=(--vault); fi
if [ "$DELTA" = "1" ]; then EXTRA+=(--delta); fi
if [ "$SHORT_CONV" = "1" ]; then EXTRA+=(--short_conv); fi
if [ "$ECHO" = "1" ]; then EXTRA+=(--echo); fi
if [ "$COND_MEM" = "1" ]; then EXTRA+=(--cond_mem); fi
if [ "$RESET_AT_EOS" = "1" ]; then EXTRA+=(--reset_at_eos); fi
if [ "$GRAD_ACCUM" != "1" ]; then EXTRA+=(--grad_accum "$GRAD_ACCUM"); fi
if [ -n "$NGRAM_ORDERS" ]; then EXTRA+=(--ngram_orders "$NGRAM_ORDERS"); fi
if [ "$ECHO_SURPRISE_TAU" != "0" ]; then EXTRA+=(--echo_surprise_tau "$ECHO_SURPRISE_TAU"); fi
if [ -n "$PAM_LAYERS" ]; then EXTRA+=(--pam_layers "$PAM_LAYERS"); fi
if [ -n "${SWA_LAYERS:-}" ]; then EXTRA+=(--swa_layers "$SWA_LAYERS" --swa_window "${SWA_WINDOW:-128}"); fi
if [ -n "${LR_SCHEDULE:-}" ]; then EXTRA+=(--lr_schedule "$LR_SCHEDULE"); fi
if [ -n "${CONTINUE_FROM:-}" ]; then EXTRA+=(--continue_from "$CONTINUE_FROM"); fi
# R2 ramp (v16/PLAN_LONGRUN.md): WEIGHTS -> RAMP_WEIGHTS over RAMP_TOKENS from cumulative RAMP_FROM_TOKENS; unset = no ramp
if [ -n "${RAMP_WEIGHTS:-}" ]; then EXTRA+=(--ramp_weights "$RAMP_WEIGHTS" --ramp_from_tokens "${RAMP_FROM_TOKENS:?}" --ramp_tokens "${RAMP_TOKENS:?}"); fi
if [ -n "${DIM:-}" ]; then EXTRA+=(--dim "$DIM"); fi
if [ -n "${N_HEADS:-}" ]; then EXTRA+=(--n_heads "$N_HEADS"); fi
if [ -n "${HEAD_DIM:-}" ]; then EXTRA+=(--head_dim "$HEAD_DIM"); fi
if [ -n "${VAL_SOURCES:-}" ]; then EXTRA+=(--val_sources "$VAL_SOURCES" --val_source_tokens "${VAL_SOURCE_TOKENS:-262144}"); fi
if [ "$MLP_HIDDEN" != "0" ]; then EXTRA+=(--mlp_hidden "$MLP_HIDDEN"); fi
if [ "$SHARE_READS" = "1" ]; then EXTRA+=(--share_reads); fi
if [ "$N_LAYERS" != "0" ]; then EXTRA+=(--n_layers "$N_LAYERS"); fi
if [ "$ANSWER_W" != "1" ]; then EXTRA+=(--answer_weight "$ANSWER_W"); fi
if [ -n "$BASE_DT_BIAS" ]; then EXTRA+=(--base_dt_bias "$BASE_DT_BIAS"); fi
if [ -n "$INIT_FROM" ]; then EXTRA+=(--init_from "$INIT_FROM"); fi
if [ "$ARCH" != "pam" ]; then EXTRA+=(--arch "$ARCH" --mamba_n_layer "$MAMBA_N_LAYER" --mamba_d_model "$MAMBA_D_MODEL"); fi
if [ -n "$TPP_GEOM" ]; then EXTRA+=(--tpp_geom "$TPP_GEOM"); fi
if [ -n "$GROVE_SPEC" ]; then EXTRA+=(--grove_spec "$GROVE_SPEC"); fi

# Auto-resume loop: --resume auto picks up <CKPT_DIR>/latest.pt when it exists
# (fresh start otherwise), restoring model/optimizer/LR schedule/step/tokens/
# best-val/RNG and the data-stream cursor. A crash (OOM, HF hiccup, box
# reboot mid-tmux) therefore costs at most SAVE_EVERY steps, not the run.
attempt=0
while :; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 1 ]; then
    echo "=== ${TAG} resume attempt $attempt $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  fi
# gdn (fla) and mamba2 (mamba_ssm 2.2.5) need packages .venv (Py3.9) lacks: default them to ~/.venvs/qllm_gdn when it exists
# (.venv's exact pins on Py3.11 + fla + mamba_ssm / causal_conv1d from qllm311, same torch 2.8.0+cu128). 2026-11-16: the ctl chain
# ran from ../qllm_ctl, whose .venv is a link to qllm311; from qllm2 the mamba2 arm would fail to import.
if [ -z "${PYTHON:-}" ] && { [ "${ARCH:-pam}" = gdn ] || [ "${ARCH:-pam}" = mamba2 ]; } && [ -x "$HOME/.venvs/qllm_gdn/bin/python" ]; then PYTHON=$HOME/.venvs/qllm_gdn/bin/python; fi
# PYTHON: the gdn arm needs fla: ~/.venvs/qllm_gdn = .venv's exact pins on Python 3.11 + fla 0.3.2 (qllm311's datasets 3.6 / pyarrow 26 deadlock the loader)
"${PYTHON:-.venv/bin/python}" -u -m v13_sempty.train \
  --preset "$PRESET" \
  --dataset mix \
  --pretrain_sources "$SOURCES" \
  --pretrain_weights "$WEIGHTS" \
  --target_tokens "$TARGET_TOKENS" \
  --blend_warmup_tokens "$BLEND_WARMUP" \
  --device cuda \
  --batch_size "$BATCH" \
  --seq_len "$SEQ" \
  --lr "$LR" \
  --warmup_steps "$WARMUP" \
  --dropout "$DROPOUT" \
  --amp_dtype bf16 \
  --fused_ce \
  --fused_pam \
  --ce_gemm_dtype auto \
  --seed "$SEED" \
  --log_interval 50 \
  --val_every "$VAL_EVERY" \
  --gen_every "$GEN_EVERY" \
  --gen_max_tokens 80 \
  --save_every_steps "$SAVE_EVERY" \
  --keep_every_steps "$KEEP_EVERY" \
  --keep_last "$KEEP_LAST" \
  --diag_every 2000 \
  --max_val_batches 0 \
  --resume auto \
  "${EXTRA[@]}" \
  --checkpoint_dir "$CKPT_DIR" >> "$LOG" 2>&1
  RC=$?
  if [ "$RC" -eq 0 ]; then break; fi
  echo "=== ${TAG} attempt $attempt died exit=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  if [ "$attempt" -ge "$MAX_RETRIES" ]; then break; fi
  if [ -f "$CKPT_DIR/.stop" ]; then      # a chain's futility rule (or a person) asked this run to stop: do not resume
    echo "=== ${TAG} stop file found: not resuming ===" | tee -a "$LOG"; break
  fi
  # A non-finite loss is a real bug, not a transient: do not resume into it.
  if tail -n 40 "$LOG" | grep -q "non-finite loss"; then
    echo "=== ${TAG} non-finite loss: not resuming ===" | tee -a "$LOG"; break
  fi
  sleep 60
done
echo "=== ${TAG} end $(date +%s) exit=$RC attempts=$attempt ===" | tee -a "$LOG"
echo "exit=$RC" > "${LOG%.log}.exit"
exit "$RC"
