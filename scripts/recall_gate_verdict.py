#!/usr/bin/env python3
"""Summarize recall-program A/B ship criteria from probe artifacts.

Reads the gate-diagnosis JSON (scripts/v11_probe_gates.py) and the behavioral
JSON (scripts/run_memory_behavioral.py) for one trained checkpoint and prints a
compact verdict against the ship criteria in the recall-program plan:

  * behavioral single-association recall at the longest context >= --acc-target
  * gate selectivity |p_content - p_filler| >= --gate-target (direction depends
    on the training sign; the recall-oriented default drives it NEGATIVE, i.e.
    the gate freezes state on low-surprisal filler and writes on content)
  * held-out PPL within --ppl-tol of the control (optional; needs both PPLs)

The unambiguous headline is behavioral recall; the gate number is diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _behavioral_summary(behavior: dict) -> dict:
    aggs = behavior.get('aggregates', [])
    if not aggs:
        return {}
    contexts = sorted({a['context_tokens'] for a in aggs})
    max_ctx = contexts[-1]
    overall = sum(a['accuracy'] for a in aggs) / len(aggs)

    def acc_where(**kw):
        rows = [a for a in aggs if all(a.get(k) == v for k, v in kw.items())]
        return (sum(r['accuracy'] for r in rows) / len(rows)) if rows else None

    single_by_ctx = {c: acc_where(context_tokens=c, associations=1) for c in contexts}
    return {
        'contexts': contexts,
        'max_context': max_ctx,
        'overall_accuracy': overall,
        'single_assoc_by_context': single_by_ctx,
        'single_assoc_at_max_context': single_by_ctx.get(max_ctx),
    }


def _gate_summary(gates: dict) -> dict:
    layers = gates.get('layers', [])
    if not layers:
        return {}
    n = len(layers)
    mean_protect = sum(l['protect_all'] for l in layers) / n
    mean_cmf = sum(l['protect_content_minus_filler'] for l in layers) / n
    mean_gamma = sum(l.get('gamma_all', 0.0) for l in layers) / n
    return {
        'n_layers': n,
        'mean_protect': mean_protect,
        'mean_content_minus_filler': mean_cmf,
        'abs_content_minus_filler': abs(mean_cmf),
        'mean_gamma': mean_gamma,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gates', type=Path, required=True)
    ap.add_argument('--behavior', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--acc-target', type=float, default=0.9)
    ap.add_argument('--gate-target', type=float, default=0.05,
                    help='min |p_content - p_filler| for a selective gate')
    ap.add_argument('--ppl', type=float, default=None, help='this arm holdout PPL')
    ap.add_argument('--baseline-ppl', type=float, default=None, help='control holdout PPL')
    ap.add_argument('--ppl-tol', type=float, default=0.02, help='allowed frac PPL regression')
    ap.add_argument('--label', default='arm')
    args = ap.parse_args()

    behavior = _load(args.behavior)
    gates = _load(args.gates)
    bsum = _behavioral_summary(behavior)
    gsum = _gate_summary(gates)

    recall = bsum.get('single_assoc_at_max_context')
    recall_pass = recall is not None and recall >= args.acc_target
    gate_pass = gsum.get('abs_content_minus_filler', 0.0) >= args.gate_target
    ppl_pass = None
    ppl_delta = None
    if args.ppl is not None and args.baseline_ppl is not None:
        ppl_delta = (args.ppl - args.baseline_ppl) / args.baseline_ppl
        ppl_pass = ppl_delta <= args.ppl_tol

    verdict = {
        'label': args.label,
        'checkpoint': gates.get('checkpoint'),
        'parameter_count': behavior.get('parameter_count'),
        'behavioral': bsum,
        'gate': gsum,
        'ppl': {'arm': args.ppl, 'baseline': args.baseline_ppl,
                'frac_delta': ppl_delta, 'tol': args.ppl_tol},
        'criteria': {
            'recall_single_at_max_context': {
                'value': recall, 'target': args.acc_target, 'pass': recall_pass},
            'gate_selectivity_abs': {
                'value': gsum.get('abs_content_minus_filler'),
                'target': args.gate_target, 'pass': gate_pass},
            'ppl_within_tol': {'value': ppl_delta, 'pass': ppl_pass},
        },
    }
    hard = [recall_pass, gate_pass] + ([ppl_pass] if ppl_pass is not None else [])
    verdict['ship'] = all(hard)

    print('=' * 64)
    print(f"RECALL-GATE VERDICT: {args.label}")
    print('=' * 64)
    print(f"  params:                  {behavior.get('parameter_count'):,}")
    print(f"  overall behavioral acc:  {bsum.get('overall_accuracy'):.3f}")
    print(f"  single-assoc by context: "
          + ', '.join(f"{c}:{(v if v is not None else float('nan')):.2f}"
                      for c, v in bsum.get('single_assoc_by_context', {}).items()))
    print(f"  single@max_ctx({bsum.get('max_context')}):     "
          f"{recall if recall is not None else float('nan'):.3f}  "
          f"target>={args.acc_target}  {'PASS' if recall_pass else 'FAIL'}")
    print(f"  gate p_content-p_filler: {gsum.get('mean_content_minus_filler'):+.3f}  "
          f"|.|>={args.gate_target}  {'PASS' if gate_pass else 'FAIL'}")
    print(f"  mean protect / gamma:    {gsum.get('mean_protect'):.3f} / {gsum.get('mean_gamma'):.3f}")
    if ppl_pass is not None:
        print(f"  holdout PPL delta:       {ppl_delta:+.3%}  tol<={args.ppl_tol:.0%}  "
              f"{'PASS' if ppl_pass else 'FAIL'}")
    print(f"  SHIP: {'YES' if verdict['ship'] else 'NO'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=2) + '\n')
        print(f"  verdict JSON -> {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
