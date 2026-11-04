"""A28 (PRE-REGISTERED 2026-11-04 08:29 IST, EXPERIMENTS_V14 "A28"): is A26's "prose held = 0.00" an artefact of the A9 read?

A26 docs (seed 26000, UNIT cell), the SAME fetched row (the package's fetch on the held session). Four reads per doc:
    a9      the package's read: held state, row + question, contrast against held state + question (no row)   [A26 'held']
    plain   held state, row + question, no contrast
    ptwin   held state + row + question, contrast against the NO-PROTOCOL session's state + the same row + question
    noprot  the A9 read on the no-protocol session                                                         [A26 'noprot']
    python v14/a28_twin.py --ckpt <weights> --wording train|a26 --lengths 2048,8192,32768 --docs 20 --out logs/...jsonl
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
def read(pkg, state, off, prefix, twin_state=None, twin_off=None, twin_prefix=None, alpha=4.0):
    la, sa, oa = pkg.mind.feed(prefix, copy.deepcopy(state), off)
    if twin_state is None:                                   # plain: alpha 0 (both streams identical)
        return pkg.tok.decode(pkg._gen(la, sa, oa, la, copy.deepcopy(sa), oa, 0.0, MAX_NEW))
    lb, sb, ob = pkg.mind.feed(twin_prefix, copy.deepcopy(twin_state), twin_off)
    return pkg.tok.decode(pkg._gen(la, sa, oa, lb, sb, ob, alpha, MAX_NEW))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--wording', default='a26', choices=['a26', 'train'])
    ap.add_argument('--lengths', default='2048,8192,32768'); ap.add_argument('--docs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=26000); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import subprocess
    print('HEADER ' + json.dumps(dict(unit='A28', hash=subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                                                      cwd=ROOT).stdout.strip(), args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device); tok = pkg.tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    fill = filler_tokens(str(FILL)); outf = open(a.out, 'a')
    for d in [int(x) for x in a.lengths.split(',')]:
        rng = random.Random(a.seed * 1000 + d + 0)                       # A26's UNIT stream (cell index 0)
        for di in range(a.docs):
            t0 = time.time()
            doc = make_doc(rng, tok, fill, 'unit', d, a.wording)
            q = EXAM.format(q=doc['question']); qi = enc(q)
            ans9, info = pkg.answer(doc['held'], q)                     # the A9 read (also fetches the row)
            row = doc['held'][info['row'][0]:info['row'][1]] if info['row'] else []
            sh = pkg.read(doc['held']); st_h, off_h = copy.deepcopy(sh['state']), sh['off']
            plain = read(pkg, st_h, off_h, row + qi)
            ansn, _ = pkg.answer(doc['noprot'], q)                      # the A9 read on the no-protocol session
            sn = pkg.read(doc['noprot']); st_n, off_n = copy.deepcopy(sn['state']), sn['off']
            ptwin = read(pkg, st_h, off_h, row + qi, st_n, off_n, row + qi)
            base = dict(d=d, doc=di, wording=a.wording, quantity=doc['q'], unit=doc['unit'], target=doc['target'],
                        proto=doc['proto'], question=doc['question'], path=info['path'], row=tok.decode(row)[-90:])
            for arm, ans in (('a9', ans9), ('plain', plain), ('ptwin', ptwin), ('noprot', ansn)):
                outf.write(json.dumps(dict(base, arm=arm, answer=ans, **score('unit', ans, doc))) + '\n')
            outf.flush()
            print(d, di, f'{time.time() - t0:.1f} s', flush=True)
    print('A28 DONE', flush=True)


if __name__ == '__main__':
    main()
