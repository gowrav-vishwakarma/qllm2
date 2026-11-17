"""CONTRAST GATE guards (PRE-REGISTERED 2026-11-17 05:39 IST + Lead's guard (d) 05:43 IST, EXPERIMENTS_V14 "CONTRAST GATE tau 0"):
real decoding of the contrast-gated told read and its composition with A32c (self-demos + split read).
Arms, same docs and fetched row:
  twin   today's told read: held vs noprot, alpha 4 on every token (Package._gen)
  split  A32c: held + self-written demos (the a32b session) vs noprot; number pieces warm (num_step), the rest alpha 4
  cgate  held vs noprot; number pieces warm; at the FIRST step after the number, GATE: t* = the alpha-4 twin's best non-stop
         token; emit t* iff (la - lb)[t*] - (la - lb)['.'] > 0 (the protocol raised t* over '.'), else '.' (bare end); after
         that, alpha 4 until a stop
  v04    the same gate on the a32b session (self-demos): the combined v0.4 candidate
Cells: train 8k / 2k, a26 2k (A28's docs, seed 26000), and the MISMATCH cell (mis): a train-wording doc whose protocol sentence
is swapped for one about the OTHER trained quantity (the asked quantity has no convention, so the right answer is the bare value).
Mismatch score: value_ok, false_unit (any mass / volume unit in the answer), right = value_ok and not false_unit.
    python v14/cgate.py --ckpt <weights> --docs 100 --out logs/v14_cgate_<tag>_<hash>_<stamp>.jsonl
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
from v14 import protocol_docs as P  # noqa: E402
from v14.a26_protocol import EXAM, FILL, MAX_NEW, item, make_doc, num_re, score, unit_re  # noqa: E402
from v14.a28_twin import read  # noqa: E402
from v14.a32b import SAMPLE_RE, find, unit_of  # noqa: E402
from v14.a32c import num_step, split_read  # noqa: E402
from v16.package import Package  # noqa: E402

ALPHA = 4.0
ALL_UNITS = [u for _, _, us in P.UNIT_Q.values() for u in us]


@torch.no_grad()
def gate_read(pkg, state, off, prefix, tstate, toff, tprefix):
    """split_read's loop; at the first non-number step the emit decision is the contrast's alone (B gets no vote)."""
    la, sa, oa = pkg.mind.feed(prefix, copy.deepcopy(state), off)
    lb, sb, ob = pkg.mind.feed(tprefix, copy.deepcopy(tstate), toff)
    out, held, gated, info = [], False, False, {}
    for _ in range(MAX_NEW):
        tw = int(la.argmax()); sofar = pkg.tok.decode(out)
        if num_step(sofar, pkg.tok.decode([tw])):
            t = tw
        else:
            tl = la + ALPHA * (la - lb)
            if not gated and re.search(r'\d+\.\d+\s*$', sofar):    # right after a complete decimal number (all targets are x.y)
                gated = True
                m = tl.clone(); m[list(pkg.stops)] = float('-inf'); ts = int(m.argmax())
                d = (la - lb).float(); g = float(d[ts] - d[pkg.dot])
                info = dict(gate_tok=pkg.tok.decode([ts]), gate_margin=g)
                t = ts if g > 0 else pkg.dot
            else:
                t = int(tl.argmax())
        if held:
            if not (t < len(pkg._digit_start) and pkg._digit_start[t]):
                break
            held = False
        out.append(t)
        if t in pkg.stops:
            if t == pkg.dot and len(out) >= 2 and out[-2] < len(pkg._digit_end) and pkg._digit_end[out[-2]] and not gated:
                held = True
            else:
                break
        la, sa, oa = pkg.mind.feed([t], sa, oa)
        lb, sb, ob = pkg.mind.feed([t], sb, ob)
    return pkg.tok.decode(out), info


def mismatch(doc, rng, enc):
    """swap the protocol sentence for one about the other trained quantity (own rng; the doc stream is untouched)."""
    oq = next(q for q in P.UNIT_Q if q != doc['q']); qp, Qp, units = P.UNIT_Q[oq]
    proto = rng.choice(P.UNIT_W).format(q=oq, qp=qp, Qp=Qp, u=rng.choice(units))
    old, new = enc(' ' + doc['proto']), enc(' ' + proto)
    p0 = find(doc['held'], old); assert p0 is not None
    return dict(doc, held=doc['held'][:p0] + new + doc['held'][p0 + len(old):], proto=proto, mis_q=oq)


def mis_score(ans, doc):
    val = re.search(num_re(doc['target']), ans) is not None
    fu = any(re.search(unit_re(u), ans) for u in ALL_UNITS)
    return dict(value_ok=int(val), false_unit=int(fu), right=int(val and not fu))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--cells', default='train:8192,mis:8192,a26:2048,train:2048,mis:2048')
    ap.add_argument('--docs', type=int, default=100); ap.add_argument('--seed', type=int, default=26000)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='CGATE guards', hash=h, args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device); tok = pkg.tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    fill = filler_tokens(str(FILL)); out = open(a.out, 'w'); summary = {}
    arms = ('twin', 'split', 'cgate', 'v04')
    for cell in a.cells.split(','):
        kind, L = cell.split(':'); L = int(L); mis = kind == 'mis'
        rng = random.Random(a.seed * 1000 + L + (7 if mis else 0))     # A28's docs; the mismatch cell has its own stream
        mrng = random.Random(a.seed * 31 + L)
        n = {k: 0 for k in arms}; fu = dict(n)
        for di in range(a.docs):
            t0 = time.time()
            doc = make_doc(rng, tok, fill, 'unit', L, 'train' if mis else kind)
            if mis:
                doc = mismatch(doc, mrng, enc)
            held, noprot = doc['held'], doc['noprot']
            sent = enc(' ' + doc['proto']); p0 = find(held, sent); assert p0 is not None
            p1 = p0 + len(sent)
            r1 = doc['rows_text'][1]; s1 = SAMPLE_RE.search(r1).group(1)
            pre = enc(' ' + r1) + enc(EXAM.format(q=doc['question'].replace(doc['sample'], s1)))
            sw, sn = pkg.read(held[:p1]), pkg.read(held[:p0])
            u_hat = unit_of(read(pkg, sw['state'], sw['off'], pre, sn['state'], sn['off'], pre), doc['q'])
            drng = random.Random(a.seed * 7919 + L * 31 + di)              # A32b / A32c's demo values
            xs = [item(drng, 'unit')[0] for _ in range(3)]
            demo = held[:p1] + enc('\nReport rule.' + ''.join(f'\n{x} -> {x} {u_hat}' for x in xs) + '\n') + held[p1:]
            qi = enc(EXAM.format(q=doc['question']))
            _, info = pkg.answer(held, EXAM.format(q=doc['question']))
            row = held[info['row'][0]:info['row'][1]] if info['row'] else []
            st_n, st_h, st_d = pkg.read(noprot), pkg.read(held), pkg.read(demo)
            pq = row + qi
            ans, gi = {}, {}
            ans['twin'] = read(pkg, st_h['state'], st_h['off'], pq, st_n['state'], st_n['off'], pq)
            ans['split'] = split_read(pkg, st_d['state'], st_d['off'], pq, st_n['state'], st_n['off'], pq)
            ans['cgate'], gi['cgate'] = gate_read(pkg, st_h['state'], st_h['off'], pq, st_n['state'], st_n['off'], pq)
            ans['v04'], gi['v04'] = gate_read(pkg, st_d['state'], st_d['off'], pq, st_n['state'], st_n['off'], pq)
            base = dict(cell=kind, L=L, doc=di, quantity=doc['q'], unit=doc['unit'], unit_hat=u_hat, target=doc['target'],
                        proto=doc['proto'], mis_q=doc.get('mis_q'), hash=h)
            for arm in arms:
                sc = mis_score(ans[arm], doc) if mis else score('unit', ans[arm], doc)
                n[arm] += sc['right']; fu[arm] += sc.get('false_unit', 0)
                out.write(json.dumps(dict(base, arm=arm, answer=ans[arm], **sc, **gi.get(arm, {}))) + '\n')
            out.flush()
            print(cell, di, f'{time.time() - t0:.1f} s', flush=True)
        d = a.docs; summary[cell] = {k: v / d for k, v in n.items()} | ({f'{k}_false_unit': v / d for k, v in fu.items()} if mis else {})
        print(f'{cell}: ' + ' '.join(f'{k} {v / d:.3f}' for k, v in n.items()) +
              (' | false_unit ' + ' '.join(f'{k} {v / d:.2f}' for k, v in fu.items()) if mis else ''), flush=True)
    print('SUMMARY ' + json.dumps(summary), flush=True)
    print('CGATE DONE', flush=True)


if __name__ == '__main__':
    main()
