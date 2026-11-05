"""A31 (PRE-REGISTERED 2026-11-05, EXPERIMENTS_V14 "A31"): state FORKS: the rule twin held from read time at O(1).

Per doc (A26 UNIT, seed 26000): stream the session up to the protocol (one prefix state), FORK it:
    A = prefix + protocol + rest,   B = prefix + rest (step offsets shifted by the protocol's length)
both forks are fed every later token; at the question, A + row + question vs B + row + question, contrast a = 4.
A28's twin read (two full reads: the held and the no-protocol session) runs beside it on the same docs; answers are
compared one by one. Reported: right, identity with A28, MB per fork (B's state), tokens re-read at question time.
    python v14/a31_fork.py --ckpt <weights> --wording train --lengths 2048 --docs 100 --out logs/...jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v14.a26_protocol import EXAM, FILL, make_doc, score  # noqa: E402
from v14.a28_twin import read  # noqa: E402
from v16.package import Package, _nbytes  # noqa: E402


def find(seq, sub):
    n = len(sub)
    for i in range(len(seq) - n + 1):
        if seq[i:i + n] == sub:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--wording', default='train', choices=['a26', 'train'])
    ap.add_argument('--lengths', default='2048'); ap.add_argument('--docs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=26000); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import subprocess
    print('HEADER ' + json.dumps(dict(unit='A31', hash=subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                                                      cwd=ROOT).stdout.strip(), args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device); tok = pkg.tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    fill = filler_tokens(str(FILL)); outf = open(a.out, 'a')
    for d in [int(x) for x in a.lengths.split(',')]:
        rng = random.Random(a.seed * 1000 + d + 0)
        for di in range(a.docs):
            t0 = time.time()
            doc = make_doc(rng, tok, fill, 'unit', d, a.wording)
            H, N = doc['held'], doc['noprot']
            prot = enc(' ' + doc['proto']); p0 = find(H, prot)
            same = int(p0 is not None and H[:p0] + H[p0 + len(prot):] == N)
            q = EXAM.format(q=doc['question']); qi = enc(q)
            _, info = pkg.answer(H, q)                                   # the package fetch (the row every read uses)
            row = H[info['row'][0]:info['row'][1]] if info['row'] else []
            # A28: two full reads at question time
            sh = pkg.read(H); st_h, off_h = copy.deepcopy(sh['state']), sh['off']
            sn = pkg.read(N); st_n, off_n = copy.deepcopy(sn['state']), sn['off']
            twin = read(pkg, st_h, off_h, row + qi, st_n, off_n, row + qi)
            # A31: one prefix, a fork at the protocol, both forks streamed to the end
            _, st_p, off_p = pkg.mind.feed(H[:p0], None, 0)
            _, st_a, off_a = pkg.mind.feed(H[p0:], copy.deepcopy(st_p), off_p)
            _, st_b, off_b = pkg.mind.feed(H[p0 + len(prot):], copy.deepcopy(st_p), off_p)
            fork = read(pkg, st_a, off_a, row + qi, st_b, off_b, row + qi)
            base = dict(d=d, doc=di, wording=a.wording, unit=doc['unit'], target=doc['target'], session_tokens=len(H),
                        proto_at=p0, proto_len=len(prot), noprot_equals_held_minus_proto=same, fork_mb=round(_nbytes(st_b) / 1e6, 3))
            outf.write(json.dumps(dict(base, arm='twin_a28', answer=twin, **score('unit', twin, doc))) + '\n')
            outf.write(json.dumps(dict(base, arm='fork_a31', answer=fork, identical=int(fork == twin), **score('unit', fork, doc))) + '\n')
            outf.flush()
            print(d, di, f'{time.time() - t0:.1f} s', flush=True)
    print('A31 DONE', flush=True)


if __name__ == '__main__':
    main()
