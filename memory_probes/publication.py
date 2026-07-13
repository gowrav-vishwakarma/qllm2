"""Reproducible multi-seed runner for the memory-probes paper.

This module intentionally keeps publication results separate from exploratory
timestamped logs. Every output contains a schema version, the exact sweep
configuration, environment metadata, raw per-seed records, and aggregates.

Examples:
    uv run python -m memory_probes.publication --profile smoke
    uv run python -m memory_probes.publication --profile mac
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from memory_probes.capacity import test_binding, test_binding_matched_bytes
from memory_probes.persistence import test_persistence
from memory_probes.rank import test_rank


SCHEMA_VERSION = 'memory-probes-publication/v1'
DEFAULT_OUTPUT = Path('logs/memory_probes/publication/cpu_results.json')


def _git_revision() -> Dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return {'commit': revision, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}


def _environment() -> Dict[str, Any]:
    return {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'git': _git_revision(),
    }


def _mean_ci95(values: Sequence[float]) -> Dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {'n': 0, 'mean': None, 'std': None, 'ci95': None}
    mean = float(data.mean())
    std = float(data.std(ddof=1)) if data.size > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(data.size) if data.size > 1 else 0.0
    return {'n': int(data.size), 'mean': mean, 'std': std, 'ci95': float(ci95)}


def _group_stats(rows: Iterable[Tuple[Tuple[Any, ...], float]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
    for key, value in rows:
        groups[key].append(float(value))
    return [
        {'key': list(key), **_mean_ci95(values)}
        for key, values in sorted(groups.items(), key=lambda item: item[0])
    ]


def _binding_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Tuple[Tuple[Any, ...], float]] = []
    for record in records:
        result = record['result']
        d = result['d']
        for method in ('matrix_pam', 'vector_hrr'):
            curve = result[method]
            rows.extend(
                ((d, method, n), accuracy)
                for n, accuracy in zip(curve['ns'], curve['accuracy'])
            )
    return _group_stats(rows)


def _matched_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Tuple[Tuple[Any, ...], float]] = []
    for record in records:
        result = record['result']
        for method in ('matrix_pam', 'vector_hrr'):
            curve = result[method]
            rows.extend(
                ((result['state_bytes'], result['matrix_d'], method, n), accuracy)
                for n, accuracy in zip(curve['ns'], curve['accuracy'])
            )
    return _group_stats(rows)


def _persistence_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Tuple[Tuple[Any, ...], float]] = []
    for record in records:
        d = record['result']['d']
        rows.extend(
            ((d, row['gamma'], row['distance']), row['relative'])
            for row in record['result']['results']
        )
    return _group_stats(rows)


def _rank_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Tuple[Tuple[Any, ...], float]] = []
    for record in records:
        result = record['result']
        rows.append(((result['d'], record['mode'], 'final'), result['ranks'][-1]))
        rows.append(((result['d'], record['mode'], 'max'), max(result['ranks'])))
    return _group_stats(rows)


def run_publication_sweep(
    *,
    seeds: Sequence[int],
    dims: Sequence[int],
    max_n: int,
    binding_trials: int,
    distances: Sequence[int],
    rank_steps: int,
    matched_dims: Sequence[int],
    matched_ns: Sequence[int],
) -> Dict[str, Any]:
    records: Dict[str, List[Dict[str, Any]]] = {
        'binding_equal_width': [],
        'binding_matched_bytes': [],
        'persistence': [],
        'rank': [],
    }

    for seed in seeds:
        for d in dims:
            records['binding_equal_width'].append({
                'seed': seed,
                'result': test_binding(
                    d=d, max_n=max_n, trials=binding_trials, seed=seed,
                ),
            })
            records['persistence'].append({
                'seed': seed,
                'result': test_persistence(
                    d=d, distances=distances, seed=seed,
                ),
            })
            for mode in ('random', 'overwrite'):
                records['rank'].append({
                    'seed': seed,
                    'mode': mode,
                    'result': test_rank(
                        d=d, steps=rank_steps, seed=seed, mode=mode,
                    ),
                })
        for matrix_d in matched_dims:
            records['binding_matched_bytes'].append({
                'seed': seed,
                'result': test_binding_matched_bytes(
                    matrix_d=matrix_d,
                    ns=matched_ns,
                    trials=binding_trials,
                    seed=seed,
                ),
            })

    aggregates = {
        'binding_equal_width': _binding_rows(records['binding_equal_width']),
        'binding_matched_bytes': _matched_rows(records['binding_matched_bytes']),
        'persistence': _persistence_rows(records['persistence']),
        'rank': _rank_rows(records['rank']),
    }
    return {
        'schema_version': SCHEMA_VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'cpu_mechanism_results',
        'environment': _environment(),
        'config': {
            'seeds': list(seeds),
            'dims': list(dims),
            'max_n': max_n,
            'binding_trials': binding_trials,
            'distances': list(distances),
            'rank_steps': rank_steps,
            'matched_dims': list(matched_dims),
            'matched_ns': list(matched_ns),
        },
        'metric_notes': {
            'accuracy': 'Top-1 identification among all values written in the trial.',
            'relative_retrieval': (
                'Absolute target alignment divided by a fresh single-write baseline; '
                'it is not a probability and can exceed one.'
            ),
            'ci95': 'Normal-approximation 95% confidence half-width across seeds.',
            'equal_width_warning': (
                'PAM uses d^2 complex state scalars while HRR uses d; use the '
                'matched-bytes result for storage-efficiency claims.'
            ),
        },
        'records': records,
        'aggregates': aggregates,
    }


def _profile(name: str) -> Dict[str, Any]:
    if name == 'smoke':
        return {
            'seeds': (42,),
            'dims': (8,),
            'max_n': 16,
            'binding_trials': 2,
            'distances': (8, 32),
            'rank_steps': 32,
            'matched_dims': (4,),
            'matched_ns': (1, 4, 8, 16),
        }
    if name == 'mac':
        return {
            'seeds': (40, 41, 42, 43, 44),
            'dims': (16, 32, 64),
            'max_n': 128,
            'binding_trials': 10,
            'distances': (64, 256, 1024),
            'rank_steps': 512,
            'matched_dims': (8, 16, 32),
            'matched_ns': (1, 4, 8, 16, 32, 64, 128),
        }
    raise ValueError(f'Unknown profile: {name}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run publication-grade CPU memory sweeps.')
    parser.add_argument('--profile', choices=('smoke', 'mac'), default='smoke')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_publication_sweep(**_profile(args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'\nPublication results saved to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
