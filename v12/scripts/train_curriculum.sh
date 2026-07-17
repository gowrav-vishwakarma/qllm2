#!/usr/bin/env bash
# V12 playable-module curriculum: grammar -> fact_retrieval -> reasoning.
#
# Each stage produces a REGISTRY MODULE (a shippable layer-group + card):
#   base (grammar): train a small DYNAMIC-HEAD grammar base from scratch
#       (v12_grammar_dyn: head_gate + high H_max => the head count AND each head's
#       phase band are LEARNED), then compact (drop closed head slots) and publish
#       as role=base (keeps shared params + its blocks).
#   fact_retrieval: load the grammar module as a frozen --substrate, grow a
#       dynamic-head DELTA "fact-band" group (error-correcting write + no-decay
#       vault + phase-address) on top, train ONLY the new layers on the
#       purpose-built masked fact data (--dataset fact) with --stage_loss ce_fact
#       (answer-masked CE + gate-surprisal + hard-negative contrastive, --fused_ce),
#       compact, publish (requires grammar, prelayer). FACT_MODE=additive A/Bs the
#       same data/loss against an additive+vault fact group.
#   reasoning: substrate = grammar + fact_retrieval, grow+freeze, --stage_loss ce,
#       compact, publish (requires grammar + fact_retrieval, prelayer).
#
# Modules land in the registry ($REGISTRY), so later stages resolve their frozen
# substrate by id@version and v12/compose.sh can assemble any inference stack.
#
# SWAPPABLE ORDER: edit ORDER to ablate acquisition order (e.g. facts first,
# grammar last). Each stage's substrate = all published predecessors in ORDER.
#
# Batch/seq target a 24GB RTX-4090 (local). On the 96GB server raise BATCH/SEQ.
#
# Usage:
#   v12/scripts/train_curriculum.sh base            # train + compact + publish grammar
#   v12/scripts/train_curriculum.sh fact_retrieval  # grow on the grammar module
#   v12/scripts/train_curriculum.sh reasoning       # grow on grammar+fact
#   v12/scripts/train_curriculum.sh all             # run every stage in ORDER
#   v12/scripts/train_curriculum.sh fact_retrieval --dry
#
# Env overrides: PY, REGISTRY, VER, AUTHOR, BATCH, SEQ, LR, TOKEN_BUDGET, HMAX,
#   GROW (e.g. "fact_retrieval:4"), STAGE_LOSS, CKPT_ROOT, DATASET/SRC/WEIGHTS,
#   FACT_MODE (delta|additive), FACT_LAYERS.
#
# 4090 smoke (validate the whole pipeline + new loader/loss before a long run):
#   BATCH=2 SEQ=1024 TOKEN_BUDGET=200000000 v12/scripts/train_curriculum.sh base
#   FACT_MODE=delta BATCH=2 SEQ=1024 TOKEN_BUDGET=150000000 \
#       v12/scripts/train_curriculum.sh fact_retrieval
#   v12/scripts/compose.sh fact_retrieval && v12/scripts/eval.sh packed_v12/model.pt
#   # A/B: FACT_MODE=additive ... fact_retrieval  (compare single_assoc@2048)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."   # repo root

PY="${PY:-.venv/bin/python}"
REGISTRY="${REGISTRY:-v12_registry}"
VER="${VER:-1.0}"
AUTHOR="${AUTHOR:-local}"
BATCH="${BATCH:-2}"          # RTX-4090 safe default
SEQ="${SEQ:-1024}"           # raise to 2048 on the server
HMAX="${HMAX:-16}"           # dynamic-head budget per grown layer (L0 prunes it)
FACT_MODE="${FACT_MODE:-delta}"   # fact-band memory: delta (error-correcting) | additive+vault (A/B)
FACT_LAYERS="${FACT_LAYERS:-4}"   # number of grown fact-band layers
CKPT_ROOT="${CKPT_ROOT:-checkpoints_v12_curriculum}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE="${1:-help}"; shift || true
DRY=0; for a in "$@"; do case "$a" in --dry|-n) DRY=1 ;; esac; done

# Skill acquisition order (edit to ablate; "base" must stay first, it is scratch).
ORDER=("base" "fact_retrieval" "reasoning")

run() { echo "+ $*"; [ "$DRY" = "1" ] && return 0; "$@"; }

# module_id published for a stage ("base" publishes as "grammar").
module_id_of() { [ "$1" = "base" ] && echo "grammar" || echo "$1"; }

# Space-separated list of ORDER entries before $1 (its substrate).
predecessors_of() {
  local target="$1" acc=()
  for s in "${ORDER[@]}"; do
    [ "$s" = "$target" ] && break
    acc+=("$s")
  done
  echo "${acc[@]:-}"
}

# Build "--substrate id@>=VER,..." from predecessors (empty for base).
substrate_arg() {
  local preds; preds="$(predecessors_of "$1")"
  [ -z "$preds" ] && { echo ""; return; }
  local list=()
  for p in $preds; do list+=("$(module_id_of "$p")@>=${VER}"); done
  local IFS=,; echo "--substrate ${list[*]}"
}

# Emit a fact-band grow-spec JSON (FACT_LAYERS entries) to $1.
#   delta   : single-state error-correcting write + vault(no-decay) + phase-addr
#   additive: K=3 additive + vault(no-decay) + phase-addr (the A/B partner)
# All entries are dynamic-head (head_gate, H_max=$HMAX) so the fact head count is
# learned then compacted. write_mode='delta' only runs at n_states=1.
write_factband_spec() {
  local out="$1" entry rest
  if [ "$FACT_MODE" = "delta" ]; then
    entry='{"skill":"fact_retrieval","group_id":"fact_retrieval","n_heads":'"$HMAX"',"n_states":1,"write_mode":"delta","delta_chunk":64,"vault_state":true,"vault_state_idx":0,"write_phase_address":true,"head_gate":true,"gate_content_aware":true}'
  else
    entry='{"skill":"fact_retrieval","group_id":"fact_retrieval","n_heads":'"$HMAX"',"n_states":3,"write_mode":"additive","vault_state":true,"vault_state_idx":0,"write_phase_address":true,"head_gate":true,"gate_content_aware":true}'
  fi
  rest=""
  for ((i=0; i<FACT_LAYERS; i++)); do
    [ -n "$rest" ] && rest="$rest,"
    rest="$rest$entry"
  done
  mkdir -p "$(dirname "$out")"
  printf '[%s]\n' "$rest" > "$out"
}

# Build "--requires id@>=VER:prelayer,..." from predecessors (empty for base).
requires_arg() {
  local preds; preds="$(predecessors_of "$1")"
  [ -z "$preds" ] && { echo ""; return; }
  local list=()
  for p in $preds; do list+=("$(module_id_of "$p")@>=${VER}:prelayer"); done
  local IFS=,; echo "${list[*]}"
}

train_stage() {
  local stage="$1"
  local ckpt_dir="$CKPT_ROOT/$stage"
  local best="$ckpt_dir/best_model.pt"
  local slim="$ckpt_dir/slim.pt"
  local mid; mid="$(module_id_of "$stage")"

  if [ "$stage" = "base" ]; then
    # Dynamic-head grammar base from scratch (few layers, grammar-heavy blend).
    run "$PY" -m v12.train \
      --preset v12_grammar_dyn --stage pretrain --dataset pretrain_mix \
      --pretrain_sources dclm,fineweb --pretrain_weights 70,30 \
      --blend_warmup_tokens 300000000 \
      --head_gate --write_phase_address \
      --stage_loss "${STAGE_LOSS:-ce}" \
      --batch_size "$BATCH" --seq_len "$SEQ" --lr "${LR:-1e-4}" --weight_decay 0.01 \
      --token_budget "${TOKEN_BUDGET:-1000000000}" \
      --checkpoint_dir "$ckpt_dir" --save_every_steps 5000
    # Compact learned head count, then publish as the stack base.
    run "$PY" -m v12.compact --checkpoint "$best" --out "$slim" --threshold 1e-3
    run "$PY" -m v12.publish --checkpoint "$slim" \
      --module_id "$mid" --version "$VER" --role base \
      --provenance "$AUTHOR" --registry "$REGISTRY" --overwrite
    return 0
  fi

  # ── Specialist stage: grow a dynamic-head group on the frozen substrate ──────
  local grow loss src weights dataset
  local -a extra=()
  case "$stage" in
    fact_retrieval)
      # Purpose-built fact stage: masked store-then-query data (answer-only loss)
      # + ce_fact (answer-masked CE + gate-surprisal + hard-negative contrastive)
      # + a grown delta/additive fact-band group. Recall aux needs the fused path.
      loss="${STAGE_LOSS:-ce_fact}"; dataset="${DATASET:-fact}"
      src="${SRC:-dclm,fineweb}"; weights="${WEIGHTS:-40,60}"
      local spec_json="$ckpt_dir/fact_band_spec.json"
      write_factband_spec "$spec_json"
      grow="${GROW:-@$spec_json}"
      extra+=(--fused_ce)
      echo "  [fact] mode=$FACT_MODE layers=$FACT_LAYERS spec=$spec_json" ;;
    reasoning)
      grow="${GROW:-reasoning:4}"; loss="${STAGE_LOSS:-ce}"
      dataset="${DATASET:-pretrain_mix}"; src="${SRC:-fineweb,smoltalk2_mid}"; weights="${WEIGHTS:-50,50}" ;;
    *)
      grow="${GROW:-$stage:3}"; loss="${STAGE_LOSS:-ce}"
      dataset="${DATASET:-pretrain_mix}"; src="${SRC:-fineweb}"; weights="${WEIGHTS:-100}" ;;
  esac

  run "$PY" -m v12.train \
    --preset v12_grammar_dyn --stage pretrain --dataset "$dataset" \
    --pretrain_sources "$src" --pretrain_weights "$weights" \
    $(substrate_arg "$stage") --registry "$REGISTRY" \
    --grow_layers "$grow" --layer_head_budget "$HMAX" \
    --head_gate --write_phase_address \
    --freeze_layers base --attach_mode "${ATTACH_MODE:-sequential}" \
    --stage_loss "$loss" "${extra[@]}" \
    --batch_size "$BATCH" --seq_len "$SEQ" --lr "${LR:-1e-4}" --weight_decay 0.01 \
    --token_budget "${TOKEN_BUDGET:-1000000000}" \
    --checkpoint_dir "$ckpt_dir" --save_every_steps 5000

  run "$PY" -m v12.compact --checkpoint "$best" --out "$slim" --threshold 1e-3
  run "$PY" -m v12.publish --checkpoint "$slim" \
    --module_id "$mid" --version "$VER" --role group --group_id "$stage" \
    --requires "$(requires_arg "$stage")" \
    --provenance "$AUTHOR" --registry "$REGISTRY" --overwrite
}

case "$STAGE" in
  all) for s in "${ORDER[@]}"; do train_stage "$s"; done ;;
  base|fact_retrieval|reasoning|*)
    if printf '%s\n' "${ORDER[@]}" | grep -qx "$STAGE"; then
      train_stage "$STAGE"
    else
      echo "Usage: $0 {all|${ORDER[*]}} [--dry]"
      echo "  base is trained from scratch; specialists grow on published predecessors."
      echo "  ORDER (edit to ablate): ${ORDER[*]}"
      exit 1
    fi ;;
esac
