"""A32b (PRE-REGISTERED 2026-11-06 15:49 IST, BUILD SPEC fixed 22:41 IST, EXPERIMENTS_V14 "A32b"): self-compiled state forks.
Right after reading a prose rule (t0), the mind's own rule-twin read decodes the unit on a DISTRACTOR row (rows_text[1], its
own question); the unit = the first alphabetic token that is not a header / quantity word. A demo block in the trained compact
format "Report rule.\n{x} -> {x} {u}" x3 (x random values, the unit written by the mind) is spliced into the session right
after the protocol. Read at the question: the protocol twin read (alpha 4) against the session WITHOUT protocol and demos.
Arms, same docs (A28's: seed 26000 * 1000 + L), same fetched row:
  twin    today's told read (held session vs noprot)
  a32b    held + self-written demos vs noprot
  oracle  held + demos with the TRUE unit vs noprot (ceiling, no bar)
Cells: held-out (a26) wording at 2k, trained wording at 8k, 100 docs each.
Bars: a26 2k a32b >= max(0.45, twin + 0.25); train 8k a32b >= max(0.55, twin + 0.25).
    python v14/a32b.py --ckpt <weights> --cells a26:2048,train:8192 --docs 100 --out logs/v14_a32b_<tag>_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v14.a26_protocol import EXAM, FILL, item, make_doc, score  # noqa: E402
from v14.a28_twin import read  # noqa: E402
from v16.package import Package  # noqa: E402

HDR = {'protocol', 'note', 'notes', 'sample', 'samples', 'report', 'rule', 'entry', 'run', 'question', 'answer', 'reading',
       'record', 'records', 'the', 'of', 'for', 'in', 'a', 'an', 'was', 'is', 'are', 'all', 'every', 'below', 'reported',
       'recorded', 'measured', 'this', 'notebook', 'log', 'unit', 'units', 'gave', 'value', 'values', 'at', 'came', 'out', 'and',
       'we', 'logged', 'had', 'which', 'what'}
SAMPLE_RE = re.compile(r'\b([A-HJ-NP-Z]\d{1,2})\b')


def unit_of(ans, q):
    """first alphabetic token that is not a header word or the quantity (spec rule; 0.89 / 0.98 on the gate jsonl)."""
    for w in re.findall(r'[^\W\d_]+', ans):
        lw = w.lower()
        if lw in HDR or lw in (q, q + 's', q + 'es'):
            continue
        return w
    return ''


def find(hay, needle):
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--cells', default='a26:2048,train:8192')
    ap.add_argument('--docs', type=int, default=100); ap.add_argument('--seed', type=int, default=26000)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='A32b', hash=h, args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device); tok = pkg.tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    fill = filler_tokens(str(FILL)); out = open(a.out, 'w'); summary = {}
    for cell in a.cells.split(','):
        wording, L = cell.split(':'); L = int(L)
        rng = random.Random(a.seed * 1000 + L + 0)                       # A28's docs (A26's UNIT stream)
        n = {'twin': 0, 'a32b': 0, 'oracle': 0}; uok = 0
        for di in range(a.docs):
            t0 = time.time()
            doc = make_doc(rng, tok, fill, 'unit', L, wording)
            held, noprot = doc['held'], doc['noprot']
            sent = enc(' ' + doc['proto']); p0 = find(held, sent); assert p0 is not None, 'protocol not found in held'
            p1 = p0 + len(sent)
            # t0: the mind's twin read on a distractor row with its own question, right after the protocol
            r1 = doc['rows_text'][1]; s1 = SAMPLE_RE.search(r1).group(1)
            q1 = doc['question'].replace(doc['sample'], s1)
            pre = enc(' ' + r1) + enc(EXAM.format(q=q1))
            sw, sn = pkg.read(held[:p1]), pkg.read(held[:p0])
            t0_ans = read(pkg, sw['state'], sw['off'], pre, sn['state'], sn['off'], pre)
            u_hat = unit_of(t0_ans, doc['q']); uok += int(u_hat == doc['unit'])
            drng = random.Random(a.seed * 7919 + L * 31 + di)              # demo values: own stream, the doc stream stays A28's
            xs = [item(drng, 'unit')[0] for _ in range(3)]
            block = lambda u: enc('\nReport rule.' + ''.join(f'\n{x} -> {x} {u}' for x in xs) + '\n')  # noqa: E731
            sessions = {'twin': held, 'a32b': held[:p1] + block(u_hat) + held[p1:],
                        'oracle': held[:p1] + block(doc['unit']) + held[p1:]}
            q = EXAM.format(q=doc['question']); qi = enc(q)
            _, info = pkg.answer(held, q)                                 # the same fetched row as A28
            row = held[info['row'][0]:info['row'][1]] if info['row'] else []
            st_n = pkg.read(noprot)
            base = dict(wording=wording, L=L, doc=di, unit=doc['unit'], unit_hat=u_hat, t0_answer=t0_ans, demos=xs,
                        target=doc['target'], proto=doc['proto'], hash=h)
            for arm, S in sessions.items():
                st = pkg.read(S)
                ans = read(pkg, st['state'], st['off'], row + qi, st_n['state'], st_n['off'], row + qi)
                sc = score('unit', ans, doc); n[arm] += sc['right']
                out.write(json.dumps(dict(base, arm=arm, answer=ans, **sc)) + '\n')
            out.flush()
            print(wording, L, di, f'unit_hat {u_hat!r} ({doc["unit"]})', f'{time.time() - t0:.1f} s', flush=True)
        d = a.docs; summary[cell] = {k: v / d for k, v in n.items()} | {'unit_hat_ok': uok / d}
        bar = max(0.45 if wording == 'a26' else 0.55, n['twin'] / d + 0.25)
        print(f"{cell}: twin {n['twin'] / d:.3f} a32b {n['a32b'] / d:.3f} oracle {n['oracle'] / d:.3f} unit_hat {uok / d:.3f}; "
              f"bar a32b >= {bar:.2f}: {'MET' if n['a32b'] / d >= bar else 'NOT met'}", flush=True)
    print('SUMMARY ' + json.dumps(summary), flush=True)
    print('A32b DONE', flush=True)


if __name__ == '__main__':
    main()
