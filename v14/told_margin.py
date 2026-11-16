"""TOLD-MARGIN diagnostic (PRE-REGISTERED 2026-11-16 20:21 IST, EXPERIMENTS_V14 "TOLD-MARGIN"): is the told unit carried in the state,
or carried but out-voted by '.' (the no-protocol answer)? One teacher-forced pass per stream, no decoding.
FAR-TOLD's docs (A28's, seed 26000, trained wording). Prefix = fetched row + question + " " + TARGET number. Next-token logits:
held (warm), noprot, twin = held + 4 (held - noprot). Units scored by their FIRST token.
    python v14/told_margin.py --ckpt <weights> --lengths 2048,8192 --docs 100 --out logs/v14_told_margin_<tag>_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v14 import protocol_docs as P  # noqa: E402
from v14.a26_protocol import EXAM, FILL, make_doc  # noqa: E402
from v16.package import Package  # noqa: E402

ALPHA = 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--lengths', default='2048,8192')
    ap.add_argument('--docs', type=int, default=100); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='TOLD-MARGIN', hash=h, args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device); tok = pkg.tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    first = {q: {u: enc(' ' + u)[0] for u in units} for q, (_, _, units) in P.UNIT_Q.items()}
    for q, m in first.items():
        assert len(set(m.values())) == len(m), f'first tokens collide within {q}: {m}'
    print('unit first tokens', {q: {u: tok.decode([t]) for u, t in m.items()} for q, m in first.items()}, flush=True)
    dot = enc('.')[0]
    fill = filler_tokens(str(FILL)); out = open(a.out, 'w')
    for L in [int(x) for x in a.lengths.split(',')]:
        rng = random.Random(26000 * 1000 + L + 0)
        acc = {k: 0 for k in ('held', 'noprot', 'twin')}; gw = dict(acc); chance = 0.0
        for di in range(a.docs):
            doc = make_doc(rng, tok, fill, 'unit', L, 'train')
            q = EXAM.format(q=doc['question']); qi = enc(q)
            _, info = pkg.answer(doc['held'], q)
            row = doc['held'][info['row'][0]:info['row'][1]] if info['row'] else []
            pre = row + qi + enc(' ' + doc['target'])
            sh, sn = pkg.read(doc['held']), pkg.read(doc['noprot'])
            lh, _, _ = pkg.mind.feed(pre, copy.deepcopy(sh['state']), sh['off'])
            ln, _, _ = pkg.mind.feed(pre, copy.deepcopy(sn['state']), sn['off'])
            lh, ln = lh.float().flatten(), ln.float().flatten()
            logits = {'held': lh, 'noprot': ln, 'twin': lh + ALPHA * (lh - ln)}
            units = first[doc['q']]; gold = units[doc['unit']]; others = [t for u, t in units.items() if u != doc['unit']]
            chance += 1 / len(units)
            rec = dict(L=L, doc=di, quantity=doc['q'], unit=doc['unit'], target=doc['target'], n_units=len(units), hash=h)
            for k, l in logits.items():
                m_id = float(l[gold] - l[others].max()); m_gate = float(l[gold] - l[dot])
                rec |= {f'{k}_unit_acc': int(m_id > 0), f'{k}_m_id': m_id, f'{k}_m_gate': m_gate, f'{k}_gate_win': int(m_gate > 0)}
                acc[k] += int(m_id > 0); gw[k] += int(m_gate > 0)
            out.write(json.dumps(rec) + '\n'); out.flush()
        d = a.docs
        print(f'L {L}: chance {chance / d:.3f} | ' + ' | '.join(f'{k} unit_acc {acc[k] / d:.2f} gate_win {gw[k] / d:.2f}' for k in acc), flush=True)
    print('TOLD-MARGIN DONE', flush=True)


if __name__ == '__main__':
    main()
