"""A32 PRE-CHECK (PRE-REGISTERED 2026-11-05, EXPERIMENTS_V14 "A32"): can the mind read a prose rule RIGHT AFTER reading it
(t0, distance ~0), which is A32's demo-writing step? Same docs as A28 / A28b at 2k (make_doc seed 26000 * 1000 + 2048), only
the distance changes: session = 256 filler tokens + the protocol + the asked row, then the A26 exam question.
  twin   = that session vs the same WITHOUT the protocol (the rule twin / fork read, alpha 4)
  plain  = that session alone (warm read)
Gate (Lead adopted): twin at t0 on the HELD-OUT wording (a26) >= 0.6 -> run A32; < 0.5 -> stop; between -> Lead decides.
    python v14/a32_precheck.py --ckpt <weights> --wordings a26,train --docs 100 --out logs/v14_a32_precheck_<tag>_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v14.a26_protocol import EXAM, FILL, make_doc, score  # noqa: E402
from v14.a28_twin import read  # noqa: E402
from v16.package import Package  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--wordings', default='a26,train')
    ap.add_argument('--docs', type=int, default=100); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=26000, help='make_doc seed base (A32b gate: 26500, fresh docs)')
    a = ap.parse_args()
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='A32 pre-check', hash=h, args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device); tok = pkg.tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    fill = filler_tokens(str(FILL)); out = open(a.out, 'w')
    for wording in a.wordings.split(','):
        rng = random.Random(a.seed * 1000 + 2048 + 0)                     # 26000 = A28's docs at 2k; 26500 = A32b's fresh docs
        n = {'twin': 0, 'plain': 0}
        for di in range(a.docs):
            doc = make_doc(rng, tok, fill, 'unit', 2048, wording)
            lead = fill[(di * 7919) % (len(fill) - 300):][:256].tolist()
            row = enc(' ' + doc['rows_text'][0]); q = enc(EXAM.format(q=doc['question']))
            with_ = lead + enc(' ' + doc['proto']); without = lead
            sw, sn = pkg.read(with_), None
            st_w, off_w = sw['state'], sw['off']
            sn = pkg.read(without); st_n, off_n = sn['state'], sn['off']
            twin = read(pkg, st_w, off_w, row + q, st_n, off_n, row + q)
            plain = read(pkg, st_w, off_w, row + q)
            for arm, ans in (('twin', twin), ('plain', plain)):
                sc = score('unit', ans, doc); n[arm] += sc['right']
                out.write(json.dumps(dict(wording=wording, doc=di, arm=arm, answer=ans, unit=doc['unit'], target=doc['target'],
                                          proto=doc['proto'], row=doc['rows_text'][0], hash=h, **sc)) + '\n')
            out.flush()
        cv = {arm: sum(json.loads(l)['conv_ok'] for l in open(a.out) if json.loads(l)['wording'] == wording and json.loads(l)['arm'] == arm)
              for arm in ('twin', 'plain')}
        print(f"{wording}: right twin {n['twin'] / a.docs:.3f} plain {n['plain'] / a.docs:.3f}; unit (conv_ok) twin "
              f"{cv['twin'] / a.docs:.3f} plain {cv['plain'] / a.docs:.3f} (n {a.docs})", flush=True)
    print('A32 PRE-CHECK DONE', flush=True)


if __name__ == '__main__':
    main()
