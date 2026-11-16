"""A32c (PRE-REGISTERED 2026-11-16 19:10 IST, EXPERIMENTS_V14 "A32c"): A32b's demos pulled their decimals into the copied value
(alpha 4 on every token). Two fixes, same docs / bodies / cells as A32b:
  twin    today's told read (held session vs noprot, alpha 4)
  a32b    held + self-written numeric demos vs noprot, alpha 4 (reference rerun)
  split   the a32b session, but the token is taken from the WARM read (alpha 0) while it continues the answer number (digits,
          one '.', the fraction digits: num_step), else from the alpha 4 contrast (the unit). Fixed 20:29 IST; the first version
          (41a7e17) took every digit / '.' piece warm, incl. the sentence-end '.', and its split rows are INVALID.
  nonum   held + number-free demos "Report rule.\\nn -> n {u}" x3 (unit written by the mind at t0) vs noprot, alpha 4
Bar: a26 2k best of split / nonum >= twin + 0.10 on BOTH bodies, and the same arm on train 8k >= a32b - 0.05.
    python v14/a32c.py --ckpt <weights> --docs 100 --out logs/v14_a32c_<tag>_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v14.a26_protocol import EXAM, FILL, MAX_NEW, item, make_doc, score  # noqa: E402
from v14.a28_twin import read  # noqa: E402
from v14.a32b import SAMPLE_RE, find, unit_of  # noqa: E402
from v16.package import Package  # noqa: E402

ALPHA = 4.0


def num_step(sofar, piece):
    """True when the warm read's next piece CONTINUES the answer number (fix 2026-11-16 20:29 IST, Argon via Lead: the first
    version took ANY digit / '.' piece warm, so the sentence-end '.' after a finished number went warm and the unit never got
    alpha 4). Allowed: digits at the start; more digits or ONE '.' after the integer; digits right after that '.'. Nothing after."""
    s = sofar.strip(); p = piece.strip() if not sofar.strip() else piece
    if not p or not all(c in '0123456789.' for c in p):
        return False
    if s == '':
        return p.isdigit()
    if re.fullmatch(r'\d+', s):
        return p.isdigit() or p == '.'
    if re.fullmatch(r'\d+\.', s):
        return p.isdigit()
    return False


@torch.no_grad()
def split_read(pkg, state, off, prefix, tstate, toff, tprefix):
    """Package._gen's greedy loop (same stops and digit-'.' hold), with the token rule of A32c's split arm."""
    la, sa, oa = pkg.mind.feed(prefix, copy.deepcopy(state), off)
    lb, sb, ob = pkg.mind.feed(tprefix, copy.deepcopy(tstate), toff)
    out, held = [], False
    for _ in range(MAX_NEW):
        tw = int(la.argmax())
        t = tw if num_step(pkg.tok.decode(out), pkg.tok.decode([tw])) else int((la + ALPHA * (la - lb)).argmax())
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
        la, sa, oa = pkg.mind.feed([t], sa, oa)
        lb, sb, ob = pkg.mind.feed([t], sb, ob)
    return pkg.tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--cells', default='a26:2048,train:8192')
    ap.add_argument('--docs', type=int, default=100); ap.add_argument('--seed', type=int, default=26000)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='A32c', hash=h, args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device); tok = pkg.tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    fill = filler_tokens(str(FILL)); out = open(a.out, 'w'); summary = {}
    arms = ('twin', 'a32b', 'split', 'nonum')
    for cell in a.cells.split(','):
        wording, L = cell.split(':'); L = int(L)
        rng = random.Random(a.seed * 1000 + L + 0)                       # A28's docs (A26's UNIT stream), as A32b
        n = dict.fromkeys(arms, 0)
        for di in range(a.docs):
            t0 = time.time()
            doc = make_doc(rng, tok, fill, 'unit', L, wording)
            held, noprot = doc['held'], doc['noprot']
            sent = enc(' ' + doc['proto']); p0 = find(held, sent); assert p0 is not None
            p1 = p0 + len(sent)
            r1 = doc['rows_text'][1]; s1 = SAMPLE_RE.search(r1).group(1)
            pre = enc(' ' + r1) + enc(EXAM.format(q=doc['question'].replace(doc['sample'], s1)))
            sw, sn = pkg.read(held[:p1]), pkg.read(held[:p0])
            t0_ans = read(pkg, sw['state'], sw['off'], pre, sn['state'], sn['off'], pre)
            u_hat = unit_of(t0_ans, doc['q'])
            drng = random.Random(a.seed * 7919 + L * 31 + di)              # A32b's demo values (same stream)
            xs = [item(drng, 'unit')[0] for _ in range(3)]
            num_block = enc('\nReport rule.' + ''.join(f'\n{x} -> {x} {u_hat}' for x in xs) + '\n')
            nonum_block = enc('\nReport rule.' + f'\nn -> n {u_hat}' * 3 + '\n')
            S = {'twin': held, 'a32b': held[:p1] + num_block + held[p1:], 'nonum': held[:p1] + nonum_block + held[p1:]}
            q = EXAM.format(q=doc['question']); qi = enc(q)
            _, info = pkg.answer(held, q)
            row = held[info['row'][0]:info['row'][1]] if info['row'] else []
            st_n = pkg.read(noprot); st = {k: pkg.read(v) for k, v in S.items()}
            ans = {k: read(pkg, st[k]['state'], st[k]['off'], row + qi, st_n['state'], st_n['off'], row + qi) for k in S}
            ans['split'] = split_read(pkg, st['a32b']['state'], st['a32b']['off'], row + qi, st_n['state'], st_n['off'], row + qi)
            base = dict(wording=wording, L=L, doc=di, unit=doc['unit'], unit_hat=u_hat, demos=xs, target=doc['target'], hash=h)
            for arm in arms:
                sc = score('unit', ans[arm], doc); n[arm] += sc['right']
                out.write(json.dumps(dict(base, arm=arm, answer=ans[arm], **sc)) + '\n')
            out.flush()
            print(wording, L, di, f'{time.time() - t0:.1f} s', flush=True)
        d = a.docs; summary[cell] = {k: v / d for k, v in n.items()}
        print(f'{cell}: ' + ' '.join(f'{k} {v / d:.3f}' for k, v in n.items()), flush=True)
    print('SUMMARY ' + json.dumps(summary), flush=True)
    print('A32c DONE', flush=True)


if __name__ == '__main__':
    main()
