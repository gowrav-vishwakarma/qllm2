#!/usr/bin/env python3
"""Compare the 50M recall+reason scale run: V13-E2b vs Transformer vs Mamba.

Reads the memory behavioral JSONs, reasoning probe JSONs, and the V13 gate probe
and prints one headline table + a Markdown block ready to paste into
v13/EXPERIMENTS_V13.md.

Usage (repo root):  .venv/bin/python scripts/compare_50m.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ARCHS = ('e2b_50m', 'transformer_50m', 'mamba_50m')
ROOT = Path(__file__).resolve().parent.parent


def mem_at(ctx: int, arch: str) -> dict:
    """Mean accuracy over the 3 target positions at each association count."""
    agg = json.load(open(ROOT / f'checkpoints_v13/{arch}/behavioral.json'))['aggregates']
    out = {}
    for a in (1, 4, 8):
        accs = [x['accuracy'] for x in agg
                if x['context_tokens'] == ctx and x['associations'] == a]
        out[a] = statistics.mean(accs) if accs else float('nan')
    return out


def reason_by_task(arch: str) -> dict:
    by: dict = {}
    for c in json.load(open(ROOT / f'checkpoints_v13/{arch}/reasoning.json'))['per_cell']:
        by.setdefault(c['task'], {})[c['gap']] = c['acc']
    return by


def gate_summary(arch: str) -> str:
    p = ROOT / f'checkpoints_v13/{arch}/gate_probe.json'
    if not p.exists():
        return '—'
    layers = json.load(open(p))['layers']
    deltas = [layer['protect_content_minus_filler'] for layer in layers
              if layer['protect_content_minus_filler'] is not None]
    if not deltas:
        return '—'
    return f'{statistics.mean(deltas):+.3f}'


def main() -> int:
    print('=' * 78)
    print('50M recall+reason scale — memory (8-way contrastive, chance 0.125)')
    print('=' * 78)
    header = f"{'arch':<16}" + ''.join(
        f'{"n" + str(n) + "@128":>10}{"n" + str(n) + "@2048":>11}' for n in (1, 4, 8))
    print(header)
    for arch in ARCHS:
        row = f'{arch:<16}'
        for ctx in (128, 2048):
            try:
                m = mem_at(ctx, arch)
                row += f'{m[1]:>10.3f}{m[4]:>11.3f}'
            except FileNotFoundError:
                row += f'{"MISSING":>21}'
        print(row)

    print()
    print('reasoning (token-exact, held-out; gap = filler sentences before answer)')
    print(f"{'arch':<16}{'task':<16}{'gap0':>7}{'gap2':>7}{'gap5':>7}")
    tasks = set()
    for arch in ARCHS:
        try:
            tasks |= set(reason_by_task(arch))
        except FileNotFoundError:
            pass
    for arch in ARCHS:
        try:
            by = reason_by_task(arch)
        except FileNotFoundError:
            print(f'{arch:<16}{"(missing reasoning.json)":<16}')
            continue
        for t in sorted(by):
            g = by[t]
            print(f'{arch:<16}{t:<16}'
                  f'{g.get(0, float("nan")):>7.2f}'
                  f'{g.get(2, float("nan")):>7.2f}'
                  f'{g.get(5, float("nan")):>7.2f}')

    print()
    print(f"{'arch':<16}{'gate Δ (content-filler, mean over layers)':>44}")
    for arch in ARCHS:
        print(f'{arch:<16}{gate_summary(arch):>44}')

    md = []
    md.append('')
    md.append('### 50M comparison (Markdown, paste-ready)')
    md.append('')
    md.append('| arch | n1@128 | n1@2048 | n4@2048 | n8@2048 | gate Δ |')
    md.append('|------|--------|---------|---------|---------|--------|')
    for arch in ARCHS:
        try:
            m128, m2048 = mem_at(128, arch), mem_at(2048, arch)
            md.append(f'| {arch} | {m128[1]:.3f} | {m2048[1]:.3f} | '
                      f'{m2048[4]:.3f} | {m2048[8]:.3f} | {gate_summary(arch)} |')
        except FileNotFoundError:
            md.append(f'| {arch} | MISSING |')
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    sys.exit(main())
