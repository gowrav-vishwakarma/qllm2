#!/usr/bin/env python3
"""Compare new memory_probes JSON output against legacy pam_math logs."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = ROOT / 'logs/v11/pam_math'
NEW_DIR = ROOT / 'logs/memory_probes'

RTOL = 1e-9
ATOL = 1e-6


def load(p: Path) -> Any:
    with p.open() as f:
        return json.load(f)


def flatten(prefix: str, obj: Any, out: Dict[str, float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return
        out[prefix] = float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            key = f'{prefix}.{k}' if prefix else k
            flatten(key, v, out)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) for x in obj):
            for i, v in enumerate(obj):
                flatten(f'{prefix}[{i}]', v, out)


def numeric_leaves(d: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    flatten('', d, out)
    return out


def compare_files(old_path: Path, new_path: Path, skip_prefixes: Tuple[str, ...] = ()) -> List[str]:
    old = load(old_path)
    new = load(new_path)

    # Normalize structure: old --all wraps in tests{}, new single-test wraps in result{}
    if 'tests' in old and 'tests' not in new:
        raise ValueError(f'{old_path.name}: expected matching --all structure')
    if 'result' in new and 'test' in new:
        new = {'tests': {new['test'].replace('-', '_'): new['result']}}
    if 'result' in new and 'test' not in new:
        new = new['result']
    if 'language_filler' in old and 'language_filler' in new:
        old_cmp = old['language_filler']
        new_cmp = new['language_filler']
    elif 'rank_text' in old and 'rank_text' in new:
        old_cmp = old['rank_text']
        new_cmp = new['rank_text']
    elif 'tests' in old and 'tests' in new:
        old_cmp, new_cmp = old['tests'], new['tests']
    elif 'result' in old:
        old_cmp = old['result']
        new_cmp = new.get('result', new)
    else:
        old_cmp, new_cmp = old, new

    o = numeric_leaves(old_cmp)
    n = numeric_leaves(new_cmp)

    issues: List[str] = []
    all_keys = sorted(set(o) | set(n))
    for k in all_keys:
        if any(k.startswith(p) or p in k for p in skip_prefixes):
            continue
        if k not in o:
            issues.append(f'  + new only: {k}={n[k]:.6g}')
            continue
        if k not in n:
            issues.append(f'  - old only: {k}={o[k]:.6g}')
            continue
        ov, nv = o[k], n[k]
        if abs(ov - nv) > ATOL + RTOL * abs(ov):
            issues.append(f'  DIFF {k}: old={ov:.8g} new={nv:.8g} delta={nv - ov:.8g}')
    return issues


def main() -> int:
    pairs = [
        ('pam_math_20260623_150756.json', 'memory_probes_all_*.json', 'full battery'),
        ('pam_math_long-context_20260623_153232.json', 'memory_probes_long_context_*.json', 'long-context 65K'),
        ('pam_math_language_filler_20260623_154049.json', 'memory_probes_language_filler_*.json', 'language filler 5K'),
        ('pam_math_rank_text_20260623_154743.json', 'memory_probes_rank_text_*.json', 'rank-text 5K'),
    ]

    if not NEW_DIR.exists() or not list(NEW_DIR.glob('*.json')):
        print('No new JSON files in', NEW_DIR)
        return 1

    ok = True
    for old_name, new_glob, label in pairs:
        old_path = OLD_DIR / old_name
        matches = sorted(NEW_DIR.glob(new_glob))
        if not old_path.exists():
            print(f'SKIP {label}: missing old {old_name}')
            continue
        if not matches:
            print(f'FAIL {label}: no new file matching {new_glob}')
            ok = False
            continue
        new_path = matches[-1]
        skip = ('elapsed_s', 'timestamp')
        issues = compare_files(old_path, new_path, skip_prefixes=skip)
        diffs = [i for i in issues if i.startswith('  DIFF')]
        extras = [i for i in issues if not i.startswith('  DIFF')]
        print(f'\n=== {label} ===')
        print(f'  old: {old_path.name}')
        print(f'  new: {new_path.name}')
        if diffs:
            ok = False
            print(f'  {len(diffs)} numeric diffs:')
            for d in diffs[:20]:
                print(d)
            if len(diffs) > 20:
                print(f'  ... and {len(diffs) - 20} more')
        else:
            print('  numeric values: MATCH')
        if extras and not diffs:
            print(f'  ({len(extras)} structural extras, no value diffs)')

    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
