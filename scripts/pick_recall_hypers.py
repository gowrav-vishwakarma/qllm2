#!/usr/bin/env python3
"""Pick best recall hyperparameters from a hypersweep root.

Ranks arms by single-assoc recall@2048, subject to PPL regression <= ppl_tol
vs baseline. Writes best_hypers.json for the from-scratch launch script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_verdict(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _val_ppl_from_log(arm: str, root_name: str) -> float | None:
    import glob
    import re
    # arm names like gate_l0.1_t1.0 -> logs may use recall_ab_gate_l0.1_t1.0_*
    pattern = f"logs/v11/recall_hypersweep/{arm}_*/v11_*.log"
    logs = sorted(glob.glob(pattern))
    if not logs:
        pattern = f"logs/v11/recall_ab_{arm}_*/v11_*.log"
        logs = sorted(glob.glob(pattern))
    if not logs:
        return None
    ppls = []
    for lp in logs:
        for line in open(lp, errors='ignore'):
            m = re.search(r'Val Loss:.*PPL:\s*([0-9.]+)', line)
            if m:
                ppls.append(float(m.group(1)))
    return ppls[-1] if ppls else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('checkpoints_v11_recall_hypersweep'))
    ap.add_argument('--baseline-ppl', type=float, default=39.75)
    ap.add_argument('--ppl-tol', type=float, default=0.05,
                    help='max fractional PPL regression allowed')
    ap.add_argument('--out', type=Path, default=Path('logs/v11/recall_hypersweep/best_hypers.json'))
    args = ap.parse_args()

    rows = []
    for arm_dir in sorted(args.root.iterdir()):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        v = _load_verdict(arm_dir / 'eval' / 'verdict.json')
        if not v:
            continue
        b = v.get('behavioral', {})
        recall = b.get('single_assoc_at_max_context')
        g = v.get('gate', {})
        ppl = _val_ppl_from_log(arm, args.root.name)
        ppl_delta = None
        ppl_ok = True
        if ppl is not None and args.baseline_ppl:
            ppl_delta = (ppl - args.baseline_ppl) / args.baseline_ppl
            ppl_ok = ppl_delta <= args.ppl_tol

        # Parse hypers from arm name
        hypers = {}
        if arm.startswith('gate_l'):
            parts = arm.replace('gate_l', '').split('_t')
            hypers = {'gate_surprisal_lambda': float(parts[0]), 'gate_surprisal_tau': float(parts[1])}
        elif arm.startswith('floor_g'):
            hypers = {'gamma_floor': float(arm.replace('floor_g', ''))}
        elif arm.startswith('recall_w'):
            hypers = {'recall_weight': int(arm.replace('recall_w', ''))}

        rows.append({
            'arm': arm,
            'recall_at_2048': recall,
            'abs_gate_selectivity': g.get('abs_content_minus_filler'),
            'val_ppl': ppl,
            'ppl_delta': ppl_delta,
            'ppl_ok': ppl_ok,
            'hypers': hypers,
        })

    eligible = [r for r in rows if r['ppl_ok'] and r['recall_at_2048'] is not None]
    eligible.sort(key=lambda x: (-x['recall_at_2048'], -(x['abs_gate_selectivity'] or 0)))

    # Pick best per category
    best_gate = max(
        (r for r in rows if r['arm'].startswith('gate_') and r['ppl_ok']),
        key=lambda x: (x['recall_at_2048'] or -1, x['abs_gate_selectivity'] or 0),
        default=None,
    )
    best_floor = max(
        (r for r in rows if r['arm'].startswith('floor_') and r['ppl_ok']),
        key=lambda x: (x['recall_at_2048'] or -1),
        default=None,
    )
    best_recall = max(
        (r for r in rows if r['arm'].startswith('recall_') and r['ppl_ok']),
        key=lambda x: (x['recall_at_2048'] or -1),
        default=None,
    )

    # Combined recommendation: gate is always on (free); floor only if PPL ok; recall from sweep
    rec = {
        'gate_surprisal_lambda': (best_gate or {}).get('hypers', {}).get('gate_surprisal_lambda', 0.1),
        'gate_surprisal_tau': (best_gate or {}).get('hypers', {}).get('gate_surprisal_tau', 1.0),
        'gate_surprisal_sign': 1.0,
        'gamma_floor': (best_floor or {}).get('hypers', {}).get('gamma_floor', 0.0),
        'recall_weight': (best_recall or {}).get('hypers', {}).get('recall_weight', 10),
        'web_weight': 96,
    }
    # If no floor passed PPL tol, disable floor
    if not best_floor:
        rec['gamma_floor'] = 0.0

    out = {
        'all_arms': rows,
        'ranked_eligible': [r['arm'] for r in eligible],
        'best_gate_arm': (best_gate or {}).get('arm'),
        'best_floor_arm': (best_floor or {}).get('arm'),
        'best_recall_arm': (best_recall or {}).get('arm'),
        'recommended': rec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
