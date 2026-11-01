"""A24 (registered 2026-11-01 14:25 IST): do rules reach the mind as DEMONSTRATIONS rather than words?

20 affine rules y = a*x + b (a in 1-3, b a trained offset), 2 held-out queries each. Arms: the rule as WORDS, as 3 or 8
demos in the trained "Map rule." format, then 2,048 tokens of held-out web text, then "\\n{x} ->"; DEMO8-NEAR = the 8
demos read 256 tokens before the query (the package's refresh). Read = v16.package.Package.apply_value with the skill
contrast (a = 4) and plain (a = 0). Free = the first answer token is " {y}".

    python v14/a24_rules.py --device cpu --out logs/v14_a24_rules_<hash>_<stamp>.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v16.package import Package  # noqa: E402

CK = {'b2p2': '/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt',
      'cu2a': '/home/gowrav/Development/qllm2/checkpoints_v13_sempty/cu2a_main_24b2969/best_model.pt'}
OFFSETS = [b for b in range(0, 21) if b not in (7, 13, 17)]
GAP, NEAR = 2048, 256


def items(seed=24001, n_rules=20, n_q=2):
    rng = random.Random(seed)
    out = []
    for r in range(n_rules):
        a, b = rng.choice((1, 2, 3)), rng.choice(OFFSETS)
        xs = rng.sample([x for x in range(0, 31) if a * x + b < 100], 8 + n_q)
        demo, qs = xs[:8], xs[8:]
        out.append(dict(rule=r, a=a, b=b, demos=demo, queries=qs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minds', default='b2p2,cu2a'); ap.add_argument('--device', default='cpu')
    ap.add_argument('--n_rules', type=int, default=20); ap.add_argument('--out', required=True)
    a_ = ap.parse_args()
    fill = filler_tokens(str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    res, rows = {}, []
    for name in a_.minds.split(','):
        pkg = Package(CK[name], device=a_.device); tok = pkg.tok
        for it in items(n_rules=a_.n_rules):
            a, b = it['a'], it['b']
            words = f"Rule: each output is {a} times the input plus {b}."
            d3 = "Map rule." + ''.join(f"\n{x} -> {a * x + b}" for x in it['demos'][:3])
            d8 = "Map rule." + ''.join(f"\n{x} -> {a * x + b}" for x in it['demos'])
            g0 = random.Random(1000 + it['rule']).randrange(0, len(fill) - GAP - 1)
            gap = tok.decode(fill[g0:g0 + GAP].tolist())
            far, near_gap = tok.decode(fill[g0:g0 + GAP - NEAR].tolist()), tok.decode(fill[g0 + GAP - NEAR:g0 + GAP].tolist())
            arms = {'WORDS': (words + '\n\n' + gap, words), 'DEMO3': (d3 + '\n\n' + gap, d3), 'DEMO8': (d8 + '\n\n' + gap, d8),
                    'DEMO8_NEAR': (far + '\n\n' + d8 + '\n\n' + near_gap, d8)}
            for x in it['queries']:
                y = a * x + b
                target = tok.encode(f' {y}', add_special_tokens=False)[0]
                for arm, (sess, pack) in arms.items():
                    for read, alpha in (('contrast', 4.0), ('plain', 0.0)):
                        ans, info = pkg.apply_value(sess, {'pack': pack, 'query': '\n{v} ->'}, str(x), alpha=alpha)
                        ok = int(info.get('first_token') == target)
                        rows.append(dict(mind=name, rule=it['rule'], a=a, b=b, x=x, y=y, arm=arm, read=read, free=ok, answer=ans))
            print(name, 'rule', it['rule'], 'done', flush=True)
        for arm in ('WORDS', 'DEMO3', 'DEMO8', 'DEMO8_NEAR'):
            for read in ('contrast', 'plain'):
                R = [r for r in rows if r['mind'] == name and r['arm'] == arm and r['read'] == read]
                res[f'{name}/{arm}/{read}'] = round(sum(r['free'] for r in R) / len(R), 3)
        print(name, {k: v for k, v in res.items() if k.startswith(name)}, flush=True)
    off = collections.defaultdict(list)
    for r in rows:
        if r['mind'] == 'b2p2' and r['arm'] == 'DEMO8' and r['read'] == 'contrast':
            off[r['b']].append(r['free'])
    res['b2p2_DEMO8_contrast_by_offset'] = {b: round(sum(v) / len(v), 2) for b, v in sorted(off.items())}
    w, d8 = res.get('b2p2/WORDS/contrast'), res.get('b2p2/DEMO8/contrast')
    if w is not None and d8 is not None:
        res['bar'] = 'MET' if (d8 >= w + 0.30 and d8 >= 0.50) else 'NOT met'
    print('BAR (b2p2 contrast: DEMO8 >= WORDS + 0.30 and >= 0.50):', res.get('bar'), flush=True)
    json.dump(dict(summary=res, rows=rows), open(a_.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
