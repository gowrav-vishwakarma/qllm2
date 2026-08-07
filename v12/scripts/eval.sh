#!/usr/bin/env bash
# Scaled evaluation for V12 checkpoints (matches the V11 eval methodology).
#
#   1. PPL on WikiText-103 val + a DCLM-edu holdout via v12.eval_checkpoints.
#   2. Dynamic-head report: learned/open head count per layer (hard-concrete gate),
#      so you can see how many heads each grammar/fact/reasoning group kept.
#   3. Behavioral recall: single_assoc@2048 (+ full grid) via v12.eval_recall, the
#      held-out key->value test the fact module targets (Mamba ~1.0 reference).
#   4. Baseline reference: a matched ~100M GPT-2 Transformer (v6/transformer_baseline.py)
#      and Mamba are the apples-to-apples references reported in V11 (train them on
#      the same pipeline for a head-to-head number).
#
# Usage:
#   v12/scripts/eval.sh ckpt1.pt [ckpt2.pt ...]
#   v12/scripts/eval.sh packed_v12/model.pt
#   LABELS=wiki,dclm SEQ=2048 BATCH=18 v12/scripts/eval.sh best_model.pt
#   RECALL=0 v12/scripts/eval.sh best_model.pt   # skip the recall eval
#
# Env: PY, LABELS, SEQ, BATCH, GATE_THRESHOLD, RECALL, RECALL_CTX, RECALL_NASSOC,
#      RECALL_TRIALS.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

PY="${PY:-.venv/bin/python}"
LABELS="${LABELS:-wiki,dclm}"
SEQ="${SEQ:-2048}"
BATCH="${BATCH:-18}"
GATE_THRESHOLD="${GATE_THRESHOLD:-1e-3}"
RECALL="${RECALL:-1}"
RECALL_CTX="${RECALL_CTX:-128,512,1024,2048}"
RECALL_NASSOC="${RECALL_NASSOC:-1,4,8}"
RECALL_TRIALS="${RECALL_TRIALS:-60}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <checkpoint> [more checkpoints ...]"
  exit 1
fi

echo "== PPL (WikiText-103 + DCLM holdout) =="
"$PY" -m v12.eval_checkpoints --checkpoints "$@" \
  --labels "$LABELS" --seq_len "$SEQ" --batch_size "$BATCH"

echo
echo "== Dynamic-head report (open heads per layer, threshold=$GATE_THRESHOLD) =="
for ckpt in "$@"; do
  echo "-- $ckpt"
  "$PY" - "$ckpt" "$GATE_THRESHOLD" <<'PY'
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
done

if [ "$RECALL" != "0" ]; then
  echo
  echo "== Behavioral recall (single_assoc@2048; held-out KEYS/VALUES) =="
  for ckpt in "$@"; do
    echo "-- $ckpt"
    "$PY" -m v12.eval_recall --checkpoint "$ckpt" \
      --context-lengths "$RECALL_CTX" --association-counts "$RECALL_NASSOC" \
      --trials "$RECALL_TRIALS"
  done
fi

echo
echo "== Baselines (references) =="
echo "  Matched ~100M GPT-2 Transformer: v6/transformer_baseline.py"
echo "  Train it on the same data pipeline (seq_len=$SEQ) for a head-to-head PPL,"
echo "  alongside the 130M Mamba reference used in the V11 comparison."
