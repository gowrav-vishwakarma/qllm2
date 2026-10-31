"""L1+P1 selector, offline over v16/bench.py JSONL with P_cands (EXPERIMENTS_V14 "L1+P1" + amendment A1).

score_i = z(sim_i) + w * z(margin_i) over the top-5 burst rows; answer = the best row's re-read answer;
abstain when that row's RAW key similarity < t. w and t are chosen on the tuning split only.

    python -m v14.l1p1_select tune  <tune kv jsonl...> --neg <tune neg jsonl...>
    python -m v14.l1p1_select score <score kv jsonl...> --neg <score neg jsonl...> --w W --t T
"""
from __future__ import annotations

import argparse
import json
import statistics as st

WS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def load(files):
    out = []
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if 'error' not in r and r.get('P_cands'):
                    out.append(r)
    return out


def z(xs):
    m = st.mean(xs)
    s = st.pstdev(xs) or 1.0
    return [(x - m) / s for x in xs]


def pick(rec, w):
    c = rec['P_cands']
    zs, zm = z([x['sim'] for x in c]), z([x['margin'] for x in c])
    i = max(range(len(c)), key=lambda j: zs[j] + w * zm[j])
    return c[i]


def kv_acc(recs, w, t):
    """accuracy with abstain-on-stored = miss; abstain rate."""
    hit = ab = 0
    for r in recs:
        b = pick(r, w)
        if b['sim'] < t:
            ab += 1
        else:
            hit += int(b['correct'])
    n = max(1, len(recs))
    return hit / n, ab / n


def neg_wrong(recs, w, t):
    """share of never-stored queries that FETCH (any fetch is wrong)."""
    return sum(int(pick(r, w)['sim'] >= t) for r in recs) / max(1, len(recs))


def tune(kv, neg):
    best = None
    for w in WS:
        acc0, _ = kv_acc(kv, w, -1.0)
        if best is None or acc0 > best[1]:
            best = (w, acc0)
    w = best[0]
    sims = sorted(pick(r, w)['sim'] for r in neg)
    t = -1.0
    for cand in [s + 1e-6 for s in sims] + [-1.0]:
        if neg_wrong(neg, w, cand) <= 0.10:
            if t == -1.0 or cand < t:
                t = cand
    acc, ab = kv_acc(kv, w, t)
    return w, t, acc, ab, neg_wrong(neg, w, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['tune', 'score'])
    ap.add_argument('kv', nargs='+')
    ap.add_argument('--neg', nargs='+', default=[])
    ap.add_argument('--w', type=float)
    ap.add_argument('--t', type=float)
    a = ap.parse_args()
    kv, neg = load(a.kv), load(a.neg)
    if a.mode == 'tune':
        w, t, acc, ab, nw = tune(kv, neg)
        print(json.dumps(dict(w=w, t=round(t, 6), tune_acc=acc, tune_abstain=ab, tune_neg_wrong_fetch=nw,
                              n_kv=len(kv), n_neg=len(neg))))
    else:
        acc, ab = kv_acc(kv, a.w, a.t)
        key_alone = sum(int(r['P_cands'][0]['correct']) for r in kv) / max(1, len(kv))
        p0 = sum(r.get('P_margin_burst', 0) for r in kv) / max(1, len(kv))
        cc = sum(r.get('P_concat_burst', 0) for r in kv) / max(1, len(kv))
        print(json.dumps(dict(P1=acc, abstain=ab, neg_wrong_fetch=neg_wrong(neg, a.w, a.t) if neg else None,
                              key_alone_reread_top1=key_alone, P0_margin=p0, concat5=cc, n_kv=len(kv), n_neg=len(neg))))


if __name__ == '__main__':
    main()
