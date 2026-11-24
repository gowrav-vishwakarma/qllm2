"""PKG-RELEASE tables (EXPERIMENTS_V14 "PKG-RELEASE"): one markdown table per runner from the jsonl rows, base vs chat side
by side. Means of the runners' own scores; no re-scoring.
    python v14/pkg_release_tables.py --suite rel=<jsonl> sft=<jsonl> --ruler rel=<jsonl> sft=<jsonl> --babilong rel=<jsonl>[,<jsonl>] sft=...
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def load(spec):
    """['rel=a.jsonl,b.jsonl', 'sft=c.jsonl'] -> {tag: [rows]}"""
    out = {}
    for s in spec or []:
        tag, paths = s.split('=', 1)
        out[tag] = [json.loads(l) for p in paths.split(',') for l in open(p) if l.strip()]
    return out


def m(xs):
    return f'{sum(xs) / len(xs):.2f}' if xs else '-'


def suite(runs):
    tags = list(runs)
    print('\n### PKG-SUITE (bare / package)\n')
    print('| cell | ' + ' | '.join(f'{t} bare | {t} package' for t in tags) + ' |')
    print('|---|' + '---|---|' * len(tags))
    rows = [('R × K, 2k / 8k (first token)', 'rxk', None), ('rewrite: told rule compiled next to the question, 2k / 8k', 'rewrite', None),
            ('told rule held in state (twin), 2k / 8k', 'told', None)]
    for name, cell, _ in rows:
        cols = []
        for t in tags:
            for arm in ('bare', 'package'):
                v = [m([r['right'] for r in runs[t] if r['cell'] == cell and r['arm'] == arm and r['L'] == L]) for L in (2048, 8192)]
                cols.append(' / '.join(v))
        print(f'| {name} | ' + ' | '.join(cols) + ' |')
    for key, name in (('free', 'recall 32k, free: needle among 8 / latest-wins / two-hop'), ('w20', 'recall 32k, 20-way: kv / kvupd / hop2')):
        cols = []
        for t in tags:
            for arm in ('bare', 'package'):
                cols.append(' / '.join(m([r[key] for r in runs[t] if r['cell'] == 'recall' and r['arm'] == arm and r['task'] == k])
                                       for k in ('kv', 'kvupd', 'hop2')))
        print(f'| {name} | ' + ' | '.join(cols) + ' |')
    for t in tags:
        mb = [r['mb'] for r in runs[t] if r['cell'] == 'recall' and r['arm'] == 'package' and isinstance(r.get('mb'), dict)]
        if mb:
            avg = {k: sum(x[k] for x in mb) / len(mb) / 1e6 for k in mb[0]}
            print(f'\n{t}: MB held per 32k recall session (mean of {len(mb)}): ' + ', '.join(f'{k} {v:.2f}' for k, v in avg.items()))


def ruler(runs):
    tags = list(runs)
    print('\n### RULER-style, our generator (bare / package; VT: package / planner chain found)\n')
    print('| L | task | n | ' + ' | '.join(f'{t} bare | {t} package' for t in tags) + ' | planner (VT, TEXT) | mind MB | library MB |')
    print('|---|---|---|' + '---|---|' * len(tags) + '---|---|---|')
    keys = sorted({(r['L'], r['task']) for t in tags for r in runs[t]})
    for L, task in keys:
        cols, n, plan, mind, lib = [], 0, [], [], []
        for t in tags:
            g = [r for r in runs[t] if r['L'] == L and r['task'] == task]
            b = [r['right'] for r in g if r['arm'] == 'bare']; p = [r for r in g if r['arm'] == 'package']
            n = len(b); cols += [m(b), m([r['right'] for r in p])]
            plan += [m([r['planner_ok'] for r in p if 'planner_ok' in r])]
            mbs = [r['mb'] for r in p if isinstance(r.get('mb'), dict)]
            if mbs:
                mind.append(sum(x['mind_state'] for x in mbs) / len(mbs) / 1e6)
                lib.append(sum(sum(v for k, v in x.items() if k not in ('mind_state', 'address_sal1')) for x in mbs) / len(mbs) / 1e6)
        tail = f' {mind[0]:.2f} | {lib[0]:.2f} |' if mind else ' - | - |'
        print(f'| {L // 1024}k | {task} | {n} | ' + ' | '.join(cols) + f" | {' / '.join(plan) if task == 'vt' else '-'} |" + tail)


def babilong(runs):
    tags = list(runs)
    print('\n### BABILong (bare = the model as is on the whole input / package / planner chain, TEXT)\n')
    print('| L | task | n | ' + ' | '.join(f'{t} bare | {t} package | {t} planner' for t in tags) + ' | mind MB | library MB |')
    print('|---|---|---|' + '---|---|---|' * len(tags) + '---|---|')
    order = ['0k', '1k', '2k', '4k', '8k', '16k', '32k']
    keys = sorted({(r['L'], r['task']) for t in tags for r in runs[t]}, key=lambda k: (order.index(k[0]), k[1]))
    for L, task in keys:
        cols, n, mind, lib = [], 0, [], []
        for t in tags:
            g = [r for r in runs[t] if r['L'] == L and r['task'] == task]
            n = max(n, len(g))
            cols += [m([r['asis_ok'] for r in g if 'asis_ok' in r]), m([r['pkg_ok'] for r in g if 'pkg_ok' in r]),
                     m([r['pkg_chain_ok'] for r in g if 'pkg_chain_ok' in r])]
            mbs = [r['mb'] for r in g if isinstance(r.get('mb'), dict)]
            if mbs:
                mind.append(sum(x.get('mind_state', 0) for x in mbs) / len(mbs) / 1e6)
                lib.append(sum(sum(v for k, v in x.items() if k != 'mind_state') for x in mbs) / len(mbs) / 1e6)
        print(f'| {L} | {task} | {n} | ' + ' | '.join(cols) + (f' | {mind[0]:.2f} | {lib[0]:.2f} |' if mind else ' | - | - |'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', nargs='*'); ap.add_argument('--ruler', nargs='*'); ap.add_argument('--babilong', nargs='*')
    a = ap.parse_args()
    if a.suite:
        suite(load(a.suite))
    if a.ruler:
        ruler(load(a.ruler))
    if a.babilong:
        babilong(load(a.babilong))


if __name__ == '__main__':
    main()
