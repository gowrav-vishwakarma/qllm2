"""FAR-TOLD secondary diagnostic (Lead / Argon 2026-11-16 19:10 IST; the headline stays the pre-registered `right`).
Many low told-8k points answer a bare "56.2." and stop. Package._gen holds a '.' after a digit for one token and DROPS that
look-ahead when it is not a digit, so a unit right after the period ("56.2. mg") is never shown. This re-reads the same docs
(A28's, seed 26000) with the protocol twin (alpha 4), greedy for 6 tokens with NO stop, and records:
  unit4   the number is followed by the unit, an early '.' between them ignored (6 free tokens)
  after_dot  the token right after the first '.'-after-the-number is the unit (what the stop rule dropped)
    python v14/fartold_unit4.py --ckpt <weights> --wording train --length 8192 --docs 100 --out logs/v14_fartold_unit4_<tag>_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v14.a26_protocol import EXAM, FILL, make_doc  # noqa: E402
from v16.package import Package  # noqa: E402


@torch.no_grad()
def free_gen(pkg, st, off, prefix, tst, toff, n=6, alpha=4.0):
    la, sa, oa = pkg.mind.feed(prefix, copy.deepcopy(st), off)
    lb, sb, ob = pkg.mind.feed(prefix, copy.deepcopy(tst), toff)
    out = []
    for _ in range(n):
        t = int((la + alpha * (la - lb)).argmax()); out.append(t)
        la, sa, oa = pkg.mind.feed([t], sa, oa); lb, sb, ob = pkg.mind.feed([t], sb, ob)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--wording', default='train'); ap.add_argument('--length', type=int, default=8192)
    ap.add_argument('--docs', type=int, default=100); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='FAR-TOLD unit4 diag', hash=h, args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device); tok = pkg.tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    fill = filler_tokens(str(FILL)); out = open(a.out, 'w')
    rng = random.Random(26000 * 1000 + a.length + 0)
    u4 = ad = 0
    for di in range(a.docs):
        doc = make_doc(rng, tok, fill, 'unit', a.length, a.wording)
        q = EXAM.format(q=doc['question']); qi = enc(q)
        _, info = pkg.answer(doc['held'], q)
        row = doc['held'][info['row'][0]:info['row'][1]] if info['row'] else []
        sh, sn = pkg.read(doc['held']), pkg.read(doc['noprot'])
        ids = free_gen(pkg, sh['state'], sh['off'], row + qi, sn['state'], sn['off'])
        pieces = [tok.decode([t]) for t in ids]
        u = doc['unit']
        text = ''.join(pieces)                           # the number, an optional early '.', then the unit (multi-token units ok)
        unit4 = int(re.match(r'\s*\d+(?:\.\d+)?\.?\s*' + re.escape(u) + r'(?![^\W\d_])', text) is not None)
        m = re.match(r'\s*\d+(?:\.\d+)?\.(?!\d)', text); after = None
        if m:                                            # the token right after the number's closing '.'
            acc = ''
            for i, p in enumerate(pieces):
                acc += p
                if len(acc) >= m.end():
                    after = ''.join(pieces[i + 1:]); break
        after_dot = int(after is not None and re.match(r'\s*' + re.escape(u) + r'(?![^\W\d_])', after) is not None)
        u4 += unit4; ad += after_dot
        out.write(json.dumps(dict(doc=di, unit=u, target=doc['target'], pieces=pieces, unit4=unit4, after_dot=after_dot,
                                  bare_dot=int(m is not None), hash=h)) + '\n'); out.flush()
    print(f'unit4 {u4 / a.docs:.3f} after_dot {ad / a.docs:.3f} (n {a.docs})', flush=True)
    print('UNIT4 DONE', flush=True)


if __name__ == '__main__':
    main()
