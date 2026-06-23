"""Unified CLI for the memory probes evaluation framework."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from memory_probes.capacity import test_binding
from memory_probes.interference import (
    test_conjugate_interference,
    test_interference,
    test_layer_bridge,
)
from memory_probes.language import test_language_filler
from memory_probes.long_context import test_long_context, test_niah, test_niah_grid
from memory_probes.persistence import test_persistence
from memory_probes.rank import test_rank, test_rank_real_text

ALL_TESTS = (
    'binding', 'persistence', 'interference', 'rank',
    'conjugate', 'layer-bridge', 'niah', 'niah-grid', 'long-context',
    'language-filler', 'rank-text',
)

DEFAULT_OUTPUT_DIR = 'logs/memory_probes'


def _parse_floats(s: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(',') if x.strip())


def _parse_ints(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(',') if x.strip())


def _output_path(out_dir: Path, tag: str) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return out_dir / f'memory_probes_{tag}_{ts}.json'


def run_all(output_dir: Path, seed: int = 42) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'seed': seed,
        'tests': {},
    }
    results['tests']['binding'] = test_binding(seed=seed)
    results['tests']['persistence'] = test_persistence(seed=seed)
    results['tests']['interference_additive'] = test_interference(seed=seed, use_delta=False)
    results['tests']['interference_delta'] = test_interference(seed=seed, use_delta=True)
    results['tests']['interference_e3'] = test_interference(seed=seed, n_states=3)
    results['tests']['rank_random'] = test_rank(seed=seed, mode='random')
    results['tests']['rank_overwrite'] = test_rank(seed=seed, mode='overwrite')
    results['tests']['conjugate'] = test_conjugate_interference(seed=seed)
    results['tests']['layer_bridge'] = test_layer_bridge(seed=seed)
    results['tests']['niah'] = test_niah(seed=seed)
    results['tests']['niah_grid'] = test_niah_grid(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = _output_path(output_dir, 'all')
    with out_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')
    return results


def run_test(test: str, args: argparse.Namespace) -> Dict[str, Any]:
    test = test.replace('_', '-')
    if test == 'binding':
        return test_binding(max_n=args.max_n, trials=args.trials, seed=args.seed)
    if test == 'persistence':
        return test_persistence(
            distances=_parse_ints(args.distances),
            gammas=_parse_floats(args.gamma),
            seed=args.seed,
        )
    if test == 'interference':
        return test_interference(
            pair_counts=(4, 8, 16, args.pairs),
            filler_counts=(0, 64, args.filler, 1024),
            seed=args.seed,
        )
    if test == 'rank':
        return test_rank(steps=args.steps, seed=args.seed)
    if test == 'conjugate':
        return test_conjugate_interference(seed=args.seed)
    if test == 'layer-bridge':
        return test_layer_bridge(modes=tuple(args.modes.split(',')), seed=args.seed)
    if test == 'niah':
        return test_niah(
            distances=_parse_ints(args.distances),
            protect_values=_parse_floats(args.protect),
            seed=args.seed,
        )
    if test == 'niah-grid':
        return test_niah_grid(
            lengths=_parse_ints(args.lengths),
            modes=tuple(args.modes.split(',')),
            seed=args.seed,
        )
    if test == 'long-context':
        return test_long_context(
            max_distance=args.max_distance,
            seed=args.seed,
            simulate_filler=not args.no_filler_sim,
        )
    if test == 'language-filler':
        return test_language_filler(
            filler_tokens=args.filler_tokens,
            gamma=args.gamma_f,
            seed=args.seed,
            projection_trials=args.projection_trials,
            projection_seed_start=args.projection_seed_start,
        )
    if test == 'rank-text':
        return test_rank_real_text(
            text_tokens=args.text_tokens,
            sample_every=args.sample_every,
            seed=args.seed,
            checkpoint=args.checkpoint or None,
            preset=args.preset,
            layer_idx=args.layer,
        )
    raise ValueError(f'Unknown test: {test}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Memory probes — recurrent matrix-memory evaluation (no checkpoint required)',
    )
    p.add_argument('--all', action='store_true', help='Run full synthetic battery')
    p.add_argument('--test', type=str, default='', help=f'One of: {",".join(ALL_TESTS)}')
    p.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--distances', type=str, default='64,128,256,512,1024,2048')
    p.add_argument('--gamma', type=str, default='0.99,0.995,0.999,1.0')
    p.add_argument('--protect', type=str, default='0,0.5,0.9,0.99,1.0')
    p.add_argument('--lengths', type=str, default='128,256,512,1024,2048')
    p.add_argument('--modes', type=str, default='baseline,e1,e2,e3')
    p.add_argument('--max-n', type=int, default=200)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--pairs', type=int, default=32)
    p.add_argument('--filler', type=int, default=256)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--max-distance', type=int, default=262144)
    p.add_argument('--no-filler-sim', action='store_true')
    p.add_argument('--filler-tokens', type=int, default=10000)
    p.add_argument('--text-tokens', type=int, default=50000)
    p.add_argument('--sample-every', type=int, default=100)
    p.add_argument('--gamma-f', type=float, default=0.995, dest='gamma_f')
    p.add_argument('--checkpoint', type=str, default='')
    p.add_argument('--preset', default='v11_e3_k3')
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--projection-trials', type=int, default=1)
    p.add_argument('--projection-seed-start', type=int, default=0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    out_dir = Path(args.output_dir)

    if args.all:
        run_all(out_dir, seed=args.seed)
        return 0

    if not args.test:
        p.print_help()
        return 1

    try:
        result = run_test(args.test, args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.test.replace('-', '_')
    out_path = _output_path(out_dir, tag)
    with out_path.open('w') as f:
        json.dump({'test': args.test, 'result': result, 'seed': args.seed}, f, indent=2)
    print(f'\nResults saved to {out_path}')
    return 0
