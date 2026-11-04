"""A30 (PRE-REGISTERED 2026-11-04, EXPERIMENTS_V14 "A30"): compositional contrast, a twin per question part.

    l = l(R,K) + a (l(R,K) − l(¬R,K)) + b (l(R,K) − l(R,¬K))
R = the protocol held in state (twin: the no-protocol session), K = the fetched row (twin: the same prefix without the row).
(a, b) = (0, 4) is the package's A9 read; (4, 0) is A28's protocol twin. A26 UNIT docs, the same fetched row.
    python v14/a30_compose.py --ckpt <weights> --wording train|a26 --lengths 2048 --docs 100 --grid 0,2,4 --seed 26000 --out ...
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v14.a26_protocol import EXAM, FILL, MAX_NEW, make_doc, score  # noqa: E402
from v16.package import Package  # noqa: E402


@torch.no_grad()
def compose(pkg, streams, a, b):
    """streams = [(state, off, prefix)] for (R,K), (¬R,K), (R,¬K); greedy over the composed logits, all streams fed the output,
    the package's stops (a '.' after a digit waits one token for a decimal)."""
    feed = pkg.mind.feed
    L, S, O = [], [], []
    for st, off, pre in streams:
        lg, s2, o2 = feed(pre, copy.deepcopy(st), off); L.append(lg); S.append(s2); O.append(o2)
    out, held = [], False
    for _ in range(MAX_NEW):
        lA, lR, lK = L
        t = int((lA + a * (lA - lR) + b * (lA - lK)).argmax())
        if held:
            if not (t < len(pkg._digit_start) and pkg._digit_start[t]):
                break
            held = False
        out.append(t)
        if t in pkg.stops:
            if t == pkg.dot and len(out) >= 2 and out[-2] < len(pkg._digit_end) and pkg._digit_end[out[-2]]:
                held = True
            else:
                break
        for i in range(3):
            L[i], S[i], O[i] = feed([t], S[i], O[i])
    return pkg.tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--wording', default='a26', choices=['a26', 'train'])
    ap.add_argument('--lengths', default='2048'); ap.add_argument('--docs', type=int, default=100)
    ap.add_argument('--grid', default='0,2,4'); ap.add_argument('--seed', type=int, default=26000)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a_ = ap.parse_args()
    import subprocess
    print('HEADER ' + json.dumps(dict(unit='A30', hash=subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                                                      cwd=ROOT).stdout.strip(), args=vars(a_))), flush=True)
    pkg = Package(a_.ckpt, device=a_.device); tok = pkg.tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    grid = [float(x) for x in a_.grid.split(',')]
    fill = filler_tokens(str(FILL)); outf = open(a_.out, 'a')
    for d in [int(x) for x in a_.lengths.split(',')]:
        rng = random.Random(a_.seed * 1000 + d + 0)
        for di in range(a_.docs):
            t0 = time.time()
            doc = make_doc(rng, tok, fill, 'unit', d, a_.wording)
            q = EXAM.format(q=doc['question']); qi = enc(q)
            _, info = pkg.answer(doc['held'], q)
            row = doc['held'][info['row'][0]:info['row'][1]] if info['row'] else []
            sh = pkg.read(doc['held']); st_h, off_h = copy.deepcopy(sh['state']), sh['off']
            sn = pkg.read(doc['noprot']); st_n, off_n = copy.deepcopy(sn['state']), sn['off']
            streams = [(st_h, off_h, row + qi), (st_n, off_n, row + qi), (st_h, off_h, qi)]
            base = dict(d=d, doc=di, wording=a_.wording, seed=a_.seed, unit=doc['unit'], target=doc['target'], path=info['path'])
            for a in grid:
                for b in grid:
                    ans = compose(pkg, streams, a, b)
                    outf.write(json.dumps(dict(base, a=a, b=b, answer=ans, **score('unit', ans, doc))) + '\n')
            outf.flush()
            print(d, di, f'{time.time() - t0:.1f} s', flush=True)
    print('A30 DONE', flush=True)


if __name__ == '__main__':
    main()
