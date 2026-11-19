"""BIND-LURE probe (Step 0 of the dual-phase proposal; PRE-REGISTERED in EXPERIMENTS_V14 "BIND-LURE Step 0"): literal-cue binding.

t40-style literal tokens, but the relation word is shared by two entities, so the cue "Name relation:" needs the conjunction:
    [filler 256] "Marta city: Lisbon.\nDaniel city: Vienna.\nRosa pet: dog.\n" (3 records, order shuffled) [filler D] "\nMarta city:"
TARGET (asked entity + relation), LURE (another entity, SAME relation), OTHER (third entity, another relation). Answers = the 8
per relation of v14/natural_recall.TYPES (first tokens distinct). Scores per doc, one teacher-forced pass at the last position:
  acc = gold beats the other 7 candidates of its relation (chance 0.125); lure = the lure's value wins; margin.
Same docs for every body: rng = Random(21000 * 1000 + D).
    python v14/bind_lure.py --ckpt <weights> --dists 64,2048 --docs 200 --device cpu --out logs/v14_bindlure_<body>_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v14.a26_protocol import FILL  # noqa: E402
from v14.natural_recall import NAMES, TYPES, first_tokens  # noqa: E402

REL = {'city': 'city', 'colour': 'car', 'day': 'lesson', 'job': 'job', 'pet': 'pet'}   # one-word relation cue per type


def make_doc(rng, enc, fill, D):
    t = rng.choice(list(TYPES)); ans = TYPES[t][0]
    n_tgt, n_lure, n_oth = rng.sample(NAMES, 3)
    a_tgt, a_lure = rng.sample(ans, 2)
    t2 = rng.choice([x for x in TYPES if x != t]); a2 = rng.choice(TYPES[t2][0])
    recs = [f'{n_tgt} {REL[t]}: {a_tgt}.', f'{n_lure} {REL[t]}: {a_lure}.', f'{n_oth} {REL[t2]}: {a2}.']
    rng.shuffle(recs)
    s0 = rng.randrange(0, len(fill) - 300 - D - 10)
    pre = [int(x) for x in fill[s0:s0 + 256]]
    gap = [int(x) for x in fill[s0 + 300:s0 + 300 + D]]
    ids = pre + enc('\n' + '\n'.join(recs) + '\n') + gap + enc(f'\n{n_tgt} {REL[t]}:')
    return dict(type=t, name=n_tgt, gold=a_tgt, lure=a_lure, ids=ids, D=D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--dists', default='64,2048')
    ap.add_argument('--docs', type=int, default=200); ap.add_argument('--device', default='cpu'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='BIND-LURE', hash=h, args=vars(a))), flush=True)
    from run_memory_behavioral import _load_v13_sempty
    from run_skillpack_probe import Body
    dev = torch.device(a.device)
    m, tok, _ = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', dev); m.eval(); body = Body(m, dev)
    print('arch', body.arch, flush=True)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    ft = first_tokens(enc); fill = filler_tokens(str(FILL)); out = open(a.out, 'w'); summ = {}
    for D in [int(x) for x in a.dists.split(',')]:
        rng = random.Random(21000 * 1000 + D); recs = []
        for di in range(a.docs):
            d = make_doc(rng, enc, fill, D); cands = ft[d['type']]
            lg = body.last_logits(d['ids'])
            cl = {w: float(lg[t]) for w, t in cands.items()}
            top = max(cl, key=cl.get)
            rec = dict(D=D, doc=di, type=d['type'], gold=d['gold'], lure=d['lure'], acc=int(top == d['gold']),
                       lure_win=int(top == d['lure']), margin=cl[d['gold']] - max(v for w, v in cl.items() if w != d['gold']),
                       n_tokens=len(d['ids']), hash=h)
            recs.append(rec); out.write(json.dumps(rec) + '\n'); out.flush()
        n = len(recs); s = dict(acc=sum(r['acc'] for r in recs) / n, lure=sum(r['lure_win'] for r in recs) / n, n=n)
        summ[D] = s
        print(f"BINDLURE D {D}: acc {s['acc']:.3f} lure {s['lure']:.3f} (n {n}, chance 0.125)", flush=True)
    print('SUMMARY ' + json.dumps(summ), flush=True)
    print('BINDLURE DONE', flush=True)


if __name__ == '__main__':
    main()
