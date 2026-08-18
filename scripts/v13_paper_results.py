#!/usr/bin/env python3
"""Summarize V13 smoke/scale results for paper update."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _recall_at_2048(behavioral: dict) -> float | None:
    for row in behavioral.get('aggregates', []):
        if row.get('context_tokens') == 2048 and row.get('associations') == 1:
            return float(row.get('accuracy', 0))
    return None


def _eff_rank_pct(rank_result: dict, head_dim: int) -> float | None:
    wiki = rank_result.get('wikitext') or rank_result.get('result', {}).get('wikitext')
    if not wiki:
        return None
    final = wiki.get('final_rank') or wiki.get('max_rank')
    if final is None:
        ranks = wiki.get('ranks', [])
        final = ranks[-1] if ranks else None
    if final is None:
        return None
    return 100.0 * float(final) / head_dim


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--v13-behavioral', type=Path, required=True)
    p.add_argument('--v13-gates', type=Path)
    p.add_argument('--v13-rank', type=Path)
    p.add_argument('--tx-behavioral', type=Path)
    p.add_argument('--out', type=Path, default=Path('memory_probes/paper/v13_results.json'))
    args = p.parse_args()

    v13_b = json.loads(args.v13_behavioral.read_text())
    out = {
        'v13_recall_2048': _recall_at_2048(v13_b),
        'v13_params': v13_b.get('parameter_count'),
    }
    if args.v13_gates and args.v13_gates.exists():
        g = json.loads(args.v13_gates.read_text())
        out['v13_gate_delta'] = g.get('mean_p_content_minus_p_filler')
    if args.v13_rank and args.v13_rank.exists():
        r = json.loads(args.v13_rank.read_text())
        result = r.get('result', r)
        head_dim = (result.get('wikitext') or {}).get('head_dim', 64)
        out['v13_eff_rank_pct'] = _eff_rank_pct(result, head_dim)
    if args.tx_behavioral and args.tx_behavioral.exists():
        tx_b = json.loads(args.tx_behavioral.read_text())
        out['transformer_recall_2048'] = _recall_at_2048(tx_b)
        out['transformer_params'] = tx_b.get('parameter_count')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
