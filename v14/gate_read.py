"""GATE (PRE-REGISTERED 2026-11-05, EXPERIMENTS_V14 "GATE"): which read should the package use, from the saved candidate
logits of the model alone (b) and the warm read with the row / examples shown (w)? Offline, CPU, no model.
  bare      argmax b                     contrast  argmax w + 4 (w - b)   (= the package today)
  warm      argmax w                     agree     argmax b if argmax b == argmax w else contrast   (no parameter; headline)
  conf      argmax b if max softmax(b) >= tau else contrast   (tau fixed on the HELD-OUT rows: best pooled accuracy)
  adacad    argmax w + a (w - b), a = 4 * JSD(softmax w, softmax b) / ln 2   (AdaCAD-style, Wang et al. 2024)
  keyed     argmax w + 4 m (w - b), m = 1 for candidates that occur (whole word) in the SHOWN span (needle: the fetched row;
            s8 / skill: the re-shown examples), else 0: the evidence overrides the mind only along its own keys (the delta
            rule's erase-at-key, which the knockouts found carries the rule, applied to the read). EXPLORATORY, needle only:
            on rule cells the answer is computed, never in the shown examples, so the mask boosts the wrong candidates
            (seen in the 2-item plumbing smoke); reported for needle, '-' elsewhere
    python v14/gate_read.py --heldout <heldout rows jsonl> --rows <pairs or pairsx jsonl> [--cells s8,needle,skill]
"""
from __future__ import annotations

import argparse
import collections
import json

import re

import numpy as np

READS = ('bare', 'warm', 'contrast', 'agree', 'conf', 'adacad', 'keyed')


def sm(x):
    e = np.exp(x - x.max()); return e / e.sum()


def picks(r, tau):
    b, w = np.asarray(r['cand_bare']), np.asarray(r['cand_warm'])
    c = w + 4 * (w - b); pb, pw = sm(b), sm(w); m = (pb + pw) / 2
    kl = lambda p: float((p * np.log((p + 1e-12) / (m + 1e-12))).sum())  # noqa: E731
    jsd = (kl(pb) + kl(pw)) / 2 / np.log(2)
    out = dict(bare=int(b.argmax()), warm=int(w.argmax()), contrast=int(c.argmax()))
    out['agree'] = out['bare'] if out['bare'] == out['warm'] else out['contrast']
    out['conf'] = out['bare'] if pb.max() >= tau else out['contrast']
    out['adacad'] = int((w + 4 * jsd * (w - b)).argmax())
    shown = r.get('package_row') if r['cell'] == 'needle' else r.get('context', '').split('\n[')[0]
    m = np.array([1.0 if shown and re.search(r'(?<![A-Za-z0-9])' + re.escape(c) + r'(?![A-Za-z0-9])', shown) else 0.0
                  for c in r['candidates']])
    out['keyed'] = int((w + 4 * m * (w - b)).argmax())
    return out


def acc(rows, tau, cells):
    g = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if r['cell'] not in cells or 'cand_bare' not in r:
            continue
        p = picks(r, tau); k = (r['cell'], r['L'] if r['cell'] != 'needle' else ('<=1k' if r['L'] <= 1024 else '2k-8k'))
        g[k]['n'] += 1
        for name in READS:
            if name == 'keyed' and r['cell'] != 'needle':
                continue
            g[k][name] += int(p[name] == r['gold_index'])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heldout', required=True); ap.add_argument('--rows', required=True)
    ap.add_argument('--cells', default='s8,needle,skill'); ap.add_argument('--out', default=None)
    a = ap.parse_args(); cells = a.cells.split(',')
    H = [json.loads(l) for l in open(a.heldout)]; R = [json.loads(l) for l in open(a.rows)]
    grid = [round(x, 2) for x in np.arange(0.30, 1.0001, 0.02)]
    pooled = {t: sum(v['conf'] for v in acc(H, t, cells).values()) / max(1, sum(v['n'] for v in acc(H, t, cells).values()))
              for t in grid}
    tau = max(grid, key=lambda t: (pooled[t], -t))
    print(f'tau (held-out, pooled conf accuracy {pooled[tau]:.4f}) = {tau}')
    res = dict(tau=tau, heldout_n=len(H), cells={})
    print('cell     L        n     ' + '  '.join(f'{x:>8s}' for x in READS))
    for k, v in sorted(acc(R, tau, cells).items(), key=lambda kv: str(kv[0])):
        show = lambda x: x != 'keyed' or k[0] == 'needle'  # noqa: E731
        print(f'{k[0]:<8s} {str(k[1]):<8s} {v["n"]:5d} ' + '  '.join(f'{v[x] / v["n"]:8.3f}' if show(x) else f'{"-":>8s}' for x in READS))
        res['cells'][f'{k[0]}@{k[1]}'] = dict(n=v['n'], **{x: round(v[x] / v['n'], 4) for x in READS if show(x)})
    tot = collections.Counter()
    for k, v in acc(R, tau, cells).items():
        if k[0] in cells:
            tot.update({x: v[x] for x in ('n',) + READS})
    print('pooled            ' + f'{tot["n"]:5d} ' + '  '.join(f'{tot[x] / tot["n"]:8.3f}' if x != 'keyed' else f'{"-":>8s}' for x in READS))
    res['pooled'] = {x: round(tot[x] / tot['n'], 4) for x in READS if x != 'keyed'}
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
