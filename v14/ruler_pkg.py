"""BENCH-PKG, RULER-STYLE part (PRE-REGISTERED 2026-11-05, EXPERIMENTS_V14 "BENCH-PKG"). OUR generator in the style of RULER
(Hsieh et al. 2024), labelled as such, not the official suite. Held-out prose as the haystack; bare vs package per doc; MB held.
  niah_single    "One of the special magic {type}s for {key} is: {value}." asked "...The special magic {type} for {key} mentioned
                 in the provided text is"; value = a word / a 7-digit number / a uuid.
  niah_multikey  the same with 3 look-alike distractor keys (same first word, other value).
  vt             "VAR A = 48213." "VAR B = VAR A." ... a chain of 4 + a distractor chain of 4; "which variables are assigned 48213?"
Bare = the model alone (state + question, greedy). Package: niah = v16/package.py answer (text fetch -> the warm read, v0.3);
vt = the library follows the chain in TEXT (labelled) and the mind writes the names from the chain rows (the package's warm read).
Right: niah = the exact value in the answer; vt = all 4 chain names and no distractor name.
    python v14/ruler_pkg.py --ckpt <weights> --lengths 4096,8192,16384,32768 --docs 50 --out logs/v14_ruler_pkg_<tag>_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import string
import sys
import time
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v7.data import _RECALL_VALUE_NOUNS  # noqa: E402
from v14.a26_protocol import FILL  # noqa: E402
from v16.package import Package  # noqa: E402

ADJ = ['lively', 'quiet', 'brisk', 'gentle', 'hollow', 'silver', 'rusty', 'calm', 'eager', 'faint', 'humble', 'jagged', 'lucky',
       'mellow', 'nimble', 'proud', 'rapid', 'sturdy', 'tender', 'vivid']
NOUN = ['otter', 'oyster', 'orchid', 'harbor', 'lantern', 'meadow', 'pebble', 'falcon', 'thimble', 'compass', 'saddle', 'kettle',
        'violin', 'glacier', 'parrot', 'quarry', 'ribbon', 'tunnel', 'walnut', 'zephyr']


def needle(kind, rng):
    if kind == 'word':
        return rng.choice(_RECALL_VALUE_NOUNS)
    if kind == 'number':
        return str(rng.randint(1_000_000, 9_999_999))
    return str(uuid.UUID(int=rng.getrandbits(128)))


def build(rng, tok, filler, L, task):
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    start = rng.randrange(0, len(filler) - L - 1); body = filler[start:start + L].tolist()
    if task in ('niah_single', 'niah_multikey'):
        kind = rng.choice(['word', 'number', 'uuid'])
        a = rng.choice(ADJ); nouns = rng.sample(NOUN, 4)
        keys = [f'{a}-{nouns[0]}'] + ([f'{a}-{n}' for n in nouns[1:]] if task == 'niah_multikey' else [])
        vals = [needle(kind, rng) for _ in keys]
        ins = [enc(f' One of the special magic {kind}s for {k} is: {v}.') for k, v in zip(keys, vals)]
        q = (f'\n\nWhat is the special magic {kind} for {keys[0]} mentioned in the provided text? '
             f'The special magic {kind} for {keys[0]} mentioned in the provided text is')
        gold = dict(value=vals[0], kind=kind, key=keys[0])
    else:                                                                   # vt
        names = [''.join(rng.choice(string.ascii_uppercase) for _ in range(5)) for _ in range(8)]
        v, dv = str(rng.randint(10000, 99999)), str(rng.randint(10000, 99999))
        chain, dchain = names[:4], names[4:]
        ins = [enc(f' VAR {chain[0]} = {v}.')] + [enc(f' VAR {chain[i]} = VAR {chain[i - 1]}.') for i in range(1, 4)]
        ins += [enc(f' VAR {dchain[0]} = {dv}.')] + [enc(f' VAR {dchain[i]} = VAR {dchain[i - 1]}.') for i in range(1, 4)]
        q = (f'\n\nFind all variables that are assigned the value {v} in the text above. Answer: According to the chain of '
             f'variable assignments in the text above, 4 variables are assigned the value {v}, they are:')
        gold = dict(value=v, names=chain, distractors=dchain)
    pos = sorted(rng.sample(range(0, max(1, L - 64)), len(ins)))
    if task == 'vt':                          # each chain keeps its own order; the two chains interleave at random slots
        slots = set(rng.sample(range(8), 4)); ia, ib = iter(ins[:4]), iter(ins[4:])
        ins = [next(ia) if s_ in slots else next(ib) for s_ in range(8)]
    else:
        rng.shuffle(ins)                      # the asked key is not always first
    out, cur = [], 0
    for p_, d in zip(pos, ins):
        out += body[cur:p_] + d; cur = p_
    out += body[cur:]
    return out, q, gold


def score(task, ans, gold):
    a = ans or ''
    if task == 'vt':
        got = set(re.findall(r'\b[A-Z]{5}\b', a))
        return int(set(gold['names']) <= got and not (got & set(gold['distractors'])))
    return int(gold['value'].lower() in a.lower())


@torch.no_grad()
def bare(pkg, sess, q_ids, max_new):
    s = pkg.read(sess); st = copy.deepcopy(s['state'])
    la, sa, oa = pkg.mind.feed(q_ids, st, s['off'])
    return pkg.tok.decode(pkg._gen(la, sa, oa, la, copy.deepcopy(sa), oa, 0.0, max_new))


def vt_chain(text, v):
    """the library follows the assignment chain in TEXT: VAR X = v, then VAR Y = VAR X, ... (returns the rows, in order)."""
    rows, names = [], []
    m = re.search(r'VAR ([A-Z]{5}) = ' + re.escape(v) + r'\.', text)
    while m:
        rows.append(m.group(0)); names.append(m.group(1))
        m = re.search(r'VAR ([A-Z]{5}) = VAR ' + names[-1] + r'\.', text)
        if len(names) > 8:
            break
    return rows, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--lengths', default='4096,8192,16384,32768')
    ap.add_argument('--docs', type=int, default=50); ap.add_argument('--tasks', default='niah_single,niah_multikey,vt')
    ap.add_argument('--seed', type=int, default=28000); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='BENCH-PKG RULER-style (our generator)', hash=h, args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device); tok = pkg.tok
    meta = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    filler = filler_tokens(str(FILL)); out = open(a.out, 'a')
    for L in [int(x) for x in a.lengths.split(',')]:
        for ti, task in enumerate(a.tasks.split(',')):
            rng = random.Random(a.seed * 1000 + L + ti)
            for di in range(a.docs):
                t0 = time.time()
                sess, q, gold = build(rng, tok, filler, L, task)
                qi = enc(q); mx = 40 if task == 'vt' else 24
                btxt = bare(pkg, sess, qi, mx)
                if task == 'vt':
                    rows, names = vt_chain(tok.decode(sess), gold['value'])
                    rows_ids = enc(' ' + ' '.join(rows)) if rows else []
                    s = pkg.read(sess)
                    la, sa, oa = pkg.mind.feed(rows_ids + qi, copy.deepcopy(s['state']), s['off'])
                    ptxt = tok.decode(pkg._gen(la, sa, oa, None, None, None, 0.0, mx))   # the package's read (v0.3 warm)
                    extra = dict(planner_ok=int(names[:4] == gold['names']), path='TEXT chain')
                    mb = dict(mind_state=s['state_bytes'], library_text=2 * len(sess))
                else:
                    ptxt, info = pkg.answer(sess, q)
                    extra = dict(path=info.get('path')); mb = info.get('bytes_held')
                base = dict(task=task, L=L, doc=di, step=meta.get('global_step'), tokens=meta.get('global_tokens'), hash=h,
                            gold=gold, mb=mb, sec=round(time.time() - t0, 1))
                out.write(json.dumps(dict(base, arm='bare', answer=btxt, right=score(task, btxt, gold))) + '\n')
                out.write(json.dumps(dict(base, arm='package', answer=ptxt, right=score(task, ptxt, gold), **extra)) + '\n')
                out.flush()
            print(L, task, 'done', flush=True)
    out.close(); summary(a.out)
    print('RULER-PKG DONE', flush=True)


def summary(path):
    """per (L, task): bare and package hit rates, the TEXT planner's chain hit (vt), MB held (mean, package rows): the mind's
    state and the library (token text + word indexes + the int8 address where the path keeps one)."""
    agg = {}
    for l in open(path):
        r = json.loads(l); k = (r['L'], r['task']); g = agg.setdefault(k, dict(bare=[], package=[], plan=[], mind=[], lib=[]))
        g[r['arm']].append(r['right'])
        if r['arm'] == 'package':
            if 'planner_ok' in r:
                g['plan'].append(r['planner_ok'])
            if isinstance(r.get('mb'), dict):                       # address_sal1 is the alternative form of the address
                b = {x: y for x, y in r['mb'].items() if x != 'address_sal1'}
                g['mind'].append(b.pop('mind_state') / 2 ** 20); g['lib'].append(sum(b.values()) / 2 ** 20)
    m = lambda x: f'{sum(x) / len(x):.2f}' if x else '-'  # noqa: E731
    print('L       task           n   bare  package  planner  mind_MB  library_MB')
    for (L, t), g in sorted(agg.items()):
        print(f'{L:<7d} {t:<14s} {len(g["bare"]):3d}  {m(g["bare"]):>5s}  {m(g["package"]):>7s}  {m(g["plan"]):>7s}  {m(g["mind"]):>7s}  {m(g["lib"])}')


if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == '--summary':
    summary(sys.argv[2]); sys.exit(0)


if __name__ == '__main__':
    main()
