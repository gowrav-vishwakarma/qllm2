"""L3 training / validation data: cross-document pairs (v14/EXPERIMENTS_V14.md "L3", pre-registered 21:19 IST).

Every sample has its stored row in document A and its question in a DIFFERENT text B (the product condition):
  synthetic : A = L tokens of pool prose with n needles " <K> ... <V>." (8 training templates, never the
              harness's); B = S tokens of other pool prose + the query template for one needle's key.
              target = the key's last token in A; qpos = the key's last token in B.
  natural   : a rare word (alphabetic, >= 5 letters, 2..40 pool occurrences) that occurs in two DIFFERENT pool
              documents; A = L tokens around the mention in one document, B = S tokens ending at the mention in
              the other. target = the mention in A; qpos = the last token of B.
Cached per sample: A's block-8 state and surprisal (rows are cut from them at training time), B's block-8 state.

    .venv/bin/python -m v14.l3_data --out <dir> --samples 3000 --seed 41 --pool_lo 0 --pool_hi 0.85
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from v7.data import _recall_key  # noqa: E402
from v14.l1_data import TEMPLATES, VALUES_TRAIN, _find, natural_phrase  # noqa: E402
from v14.pack import Mind  # noqa: E402

POOL = str(ROOT / '.cache/v14_train_pool_fineweb014.pt')
EOS = 50256


def load_pool(lo, hi):
    t = torch.load(POOL)['tokens'].numpy()
    t = t[int(lo * len(t)):int(hi * len(t))]
    doc = np.cumsum(t == EOS)                       # document index of every token
    keep = (t != EOS) & (t != 198) & (t != 628)
    return t[keep], doc[keep]


def synthetic(rng, tok, pool, L, S):
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    a0 = rng.randrange(0, len(pool) - L - 1)
    body = pool[a0:a0 + L].tolist()
    n = rng.randint(6, 24)
    use_nat = rng.random() < 0.5
    ks, seen = [], set()
    while len(ks) < n:
        k = natural_phrase(rng, tok, pool, (a0, a0 + L)) if use_nat else _recall_key(rng)
        if k not in seen:
            seen.add(k)
            ks.append(k)
    tpl = rng.choice(TEMPLATES)
    pos = sorted(rng.sample(range(0, L - 64), n))
    A, cur, starts = [], 0, []
    for p, k in zip(pos, ks):
        A += body[cur:p]
        cur = p
        starts.append(len(A))
        A += enc(tpl[0].format(K=k, V=rng.choice(VALUES_TRAIN)))
    A += body[cur:]
    qi = rng.randrange(n)
    kid = enc(' ' + ks[qi])
    t = _find(A, kid, starts[qi])
    while True:                                      # the session comes from a region away from A
        b0 = rng.randrange(0, len(pool) - S - 1)
        if b0 + S < a0 or b0 > a0 + L:
            break
    q = enc(tpl[1].format(K=ks[qi]))
    B = pool[b0:b0 + S].tolist() + q
    qp = _find(B, kid, S)
    if t is None or qp is None:
        return None
    return dict(A=A, B=B, target=t + len(kid) - 1, qpos=qp + len(kid) - 1, kind='synthetic')


def natural(rng, tok, pool, doc, L, S, index):
    for _ in range(50):
        w = rng.choice(index['words'])
        occ = index['occ'][w]
        by_doc = defaultdict(list)
        for p in occ:
            by_doc[int(doc[p])].append(p)
        if len(by_doc) < 2:
            continue
        d1, d2 = rng.sample(sorted(by_doc), 2)
        pa, pb = rng.choice(by_doc[d1]), rng.choice(by_doc[d2])
        a0 = max(0, min(len(pool) - L, pa - rng.randrange(64, L - 64)))
        b0 = pb - S + 1
        if b0 < 0 or not (b0 + S <= a0 or b0 >= a0 + L):
            continue
        return dict(A=pool[a0:a0 + L].tolist(), B=pool[b0:pb + 1].tolist(), target=pa - a0, qpos=S - 1, kind='natural')
    return None


def rare_index(tok, pool):
    counts = np.bincount(pool, minlength=50257)
    ok = np.zeros(50257, dtype=bool)
    for t in np.nonzero((counts >= 2) & (counts <= 40))[0]:
        s = tok.decode([int(t)])
        ok[t] = s.startswith(' ') and s.strip().isalpha() and len(s.strip()) >= 5
    occ = defaultdict(list)
    for p in np.nonzero(ok[pool])[0]:
        occ[int(pool[p])].append(int(p))
    words = [w for w, ps in occ.items() if len(ps) >= 2]
    return dict(words=words, occ=occ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=41)
    ap.add_argument('--length', type=int, default=4096)
    ap.add_argument('--session', type=int, default=1024)
    ap.add_argument('--pool_lo', type=float, default=0.0)
    ap.add_argument('--pool_hi', type=float, default=0.85)
    ap.add_argument('--frac_natural', type=float, default=0.5)
    ap.add_argument('--mem_frac', type=float, default=0.2)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    mind = Mind()
    tok = mind.tok
    pool, doc = load_pool(a.pool_lo, a.pool_hi)
    index = rare_index(tok, pool)
    print(f"pool {len(pool):,} tokens, {len(index['words']):,} rare words with >= 2 mentions", flush=True)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made = 0
    while made < a.samples:
        s = natural(rng, tok, pool, doc, a.length, a.session, index) if rng.random() < a.frac_natural \
            else synthetic(rng, tok, pool, a.length, a.session)
        if s is None:
            continue
        va, nll = mind.read(s['A'])
        vb, _ = mind.read(s['B'])
        torch.save(dict(ids=s['A'], vecs=va.half().cpu(), nll=nll.half().cpu(), qvecs=vb.half().cpu(),
                        target=s['target'], qpos=s['qpos'], tlen=1, kind=s['kind'], cross=True),
                   out / f'doc_{made:05d}.pt')
        made += 1
        if made % 200 == 0:
            print(f"  {made}/{a.samples}", flush=True)
    print(f"saved {made} samples to {out}", flush=True)


if __name__ == '__main__':
    main()
