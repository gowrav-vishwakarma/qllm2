"""Deprecated: use `python -m memory_probes --test language-filler` or `--test rank-text`.

This module re-exports language/rank probes for backward compatibility.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from memory_probes.cli import DEFAULT_OUTPUT_DIR, _output_path
from memory_probes.language import test_language_filler
from memory_probes.rank import test_rank_real_text

warnings.warn(
    'v11.pam_math_language is deprecated; use `python -m memory_probes`',
    DeprecationWarning,
    stacklevel=2,
)

TESTS = ('language-filler', 'rank-text', 'both')


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description='Deprecated — use python -m memory_probes')
    p.add_argument('--test', default='both', choices=TESTS)
    p.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--filler-tokens', type=int, default=10000)
    p.add_argument('--text-tokens', type=int, default=50000)
    p.add_argument('--sample-every', type=int, default=100)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--checkpoint', type=str, default='')
    p.add_argument('--preset', default='v11_e3_k3')
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--projection-trials', type=int, default=1)
    p.add_argument('--projection-seed-start', type=int, default=0)
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {'timestamp': datetime.now(timezone.utc).isoformat(), 'seed': args.seed}

    if args.test in ('language-filler', 'both'):
        results['language_filler'] = test_language_filler(
            filler_tokens=args.filler_tokens,
            gamma=args.gamma,
            seed=args.seed,
            projection_trials=args.projection_trials,
            projection_seed_start=args.projection_seed_start,
        )

    if args.test in ('rank-text', 'both'):
        results['rank_text'] = test_rank_real_text(
            text_tokens=args.text_tokens,
            sample_every=args.sample_every,
            seed=args.seed,
            checkpoint=args.checkpoint or None,
            preset=args.preset,
            layer_idx=args.layer,
        )

    tag = args.test.replace('-', '_')
    out_path = _output_path(out_dir, tag)
    with out_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
