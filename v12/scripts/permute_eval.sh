#!/usr/bin/env bash
# Permutation eval: score each packed stack on PPL, fact recall, head-gate
# report, and a few fixed generation probes.
#
# Usage:
#   v12/scripts/permute_eval.sh [arm1 arm2 ...]
#   ARMS="gfr grf" SEQ=1024 BATCH=4 RECALL_TRIALS=30 v12/scripts/permute_eval.sh
#
# Env:
#   PY, REGISTRY, LOGDIR, ARMS (default all 5), SEQ, BATCH, RECALL_TRIALS,
#   GEN_TOKENS, GEN_TEMPERATURE, GEN_SEED.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

PY="${PY:-.venv/bin/python}"
LOGDIR="${LOGDIR:-logs/v12_perm}"
SEQ="${SEQ:-1024}"
BATCH="${BATCH:-4}"
RECALL_TRIALS="${RECALL_TRIALS:-30}"
GEN_TOKENS="${GEN_TOKENS:-80}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.7}"
GEN_SEED="${GEN_SEED:-7}"
GATE_THRESHOLD="${GATE_THRESHOLD:-1e-3}"

mkdir -p "$LOGDIR"

# arm -> packed checkpoint path
declare -A PACK=(
  [gfr]="packed_v12/gfr.pt"
  [grf]="packed_v12/grf.pt"
  [gf11r]="packed_v12/gf11r.pt"
  [gf]="packed_v12/gf.pt"
  [gr]="packed_v12/gr.pt"
)

if [ "$#" -gt 0 ]; then
  ARMS="$*"
elif [ -z "${ARMS:-}" ]; then
  ARMS="gfr grf gf11r gf gr"
fi

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" | tee -a "$LOGDIR/sweep.log"; }

# Fixed generation probes (one per skill surface).
FACT_PROMPT="Record: bofim means gold. Record: kaner means silver. Query: bofim means"
REASON_PROMPT="Question: If a train travels 60 km in 2 hours, its speed is"
GRAMMAR_PROMPT="In 1923 , the University of"

run_ppl() {
  local ckpt="$1" out="$2"
  "$PY" -m v12.eval_checkpoints --checkpoints "$ckpt" \
    --labels wiki,dclm --seq_len "$SEQ" --batch_size "$BATCH" \
    > "$out" 2>&1
}

run_recall() {
  local ckpt="$1" out="$2"
  "$PY" -m v12.eval_recall --checkpoint "$ckpt" \
    --context-lengths 128,512,1024 \
    --association-counts 1,4,8 \
    --trials "$RECALL_TRIALS" \
    --output "$out" 2>&1 | tee "${out%.json}.log"
}

run_headgate() {
  local ckpt="$1" out="$2"
  "$PY" - "$ckpt" "$GATE_THRESHOLD" >"$out" 2>&1 <<'PY'
import sys, torch
from v12.model import V12Config
from v12.compact import _hard_concrete_z
ckpt_path, thr = sys.argv[1], float(sys.argv[2])
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = ck['config']; specs = cfg.get('layer_specs') or []
state = ck['model_state_dict']
n_layers = cfg.get('n_layers', len(specs))
for i in range(n_layers):
    key = f'blocks.{i}.pam.head_gate.log_alpha'
    grp = specs[i].get('group_id') if i < len(specs) else '?'
    if key in state:
        z = _hard_concrete_z(state[key])
        open_h = int((z > thr).sum()); total = z.numel()
        print(f"  layer {i:>2} [{grp:<14}] heads open: {open_h}/{total}")
    else:
        print(f"  layer {i:>2} [{grp:<14}] (no head gate; fixed heads)")
PY
}

run_gen() {
  local ckpt="$1" out="$2" prompt="$3"
  "$PY" -m v12.generate --checkpoint "$ckpt" --prompt "$prompt" \
    --max_tokens "$GEN_TOKENS" --temperature "$GEN_TEMPERATURE" \
    --top_k 50 --top_p 0.9 --repetition_penalty 1.2 --seed "$GEN_SEED" \
    > "$out" 2>&1
}

eval_arm() {
  local arm="$1" ckpt="${PACK[$arm]}"
  local dir="$LOGDIR/$arm"
  mkdir -p "$dir"
  log "=== arm $arm ($ckpt) START ==="

  log "[$arm] PPL"
  run_ppl "$ckpt" "$dir/ppl.log" || log "[$arm] PPL FAILED (see $dir/ppl.log)"

  log "[$arm] recall"
  run_recall "$ckpt" "$dir/recall.json" || log "[$arm] recall FAILED"

  log "[$arm] head-gate"
  run_headgate "$ckpt" "$dir/headgate.log" || log "[$arm] head-gate FAILED"

  log "[$arm] gen (fact)"
  run_gen "$ckpt" "$dir/gen_fact.txt" "$FACT_PROMPT" || log "[$arm] gen_fact FAILED"
  log "[$arm] gen (reasoning)"
  run_gen "$ckpt" "$dir/gen_reasoning.txt" "$REASON_PROMPT" || log "[$arm] gen_reasoning FAILED"
  log "[$arm] gen (grammar)"
  run_gen "$ckpt" "$dir/gen_grammar.txt" "$GRAMMAR_PROMPT" || log "[$arm] gen_grammar FAILED"

  log "=== arm $arm DONE ==="
}

log "=== permutation sweep START (arms: $ARMS) seq=$SEQ batch=$BATCH trials=$RECALL_TRIALS ==="
for arm in $ARMS; do
  if [ -z "${PACK[$arm]:-}" ]; then
    log "WARN: unknown arm '$arm' (skip)"
    continue
  fi
  if [ ! -f "${PACK[$arm]}" ]; then
    log "WARN: pack missing for arm '$arm' at ${PACK[$arm]} (skip)"
    continue
  fi
  eval_arm "$arm"
done
log "=== permutation sweep COMPLETE ==="
log "summary: $LOGDIR/sweep_summary.txt (built separately)"
