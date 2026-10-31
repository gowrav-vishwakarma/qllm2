"""L2 two-organ fetch, offline over v16/bench.py JSONL (EXPERIMENTS_V14 "L2", pre-registered 21:12 IST).

Per query: exact organ hit (B_hit) -> its answer (B_ngram); else the learned key's top burst row re-read
(P_cands[0]); abstain when that row's raw key similarity < t. t is tuned only on the tuning split.

    python -m v14.l2_select tune  --kv <tune kv jsonl...> --neg <tune neg jsonl...>
    python -m v14.l2_select score --kv <jsonl...> [--neg <jsonl...>] --t T
"""
from __future__ import annotations

import argparse
import json


def load(files):
    out = []
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if 'error' not in r and r.get('P_cands'):
                    out.append(r)
    return out


def decide(r, t):
    """-> ('organ' | 'key' | 'abstain', correct)."""
    if r.get('B_hit'):
        return 'organ', int(r.get('B_ngram', 0))
    top = r['P_cands'][0]
    if top['sim'] < t:
        return 'abstain', 0
    return 'key', int(top['correct'])


def acc(recs, t):
    d = [decide(r, t) for r in recs]
    n = max(1, len(d))
    return dict(acc=sum(c for _, c in d) / n, abstain=sum(k == 'abstain' for k, _ in d) / n,
                organ=sum(k == 'organ' for k, _ in d) / n, n=len(d))


def neg_wrong(recs, t):
    return sum(int(r.get('B_hit') or r['P_cands'][0]['sim'] >= t) for r in recs) / max(1, len(recs))


def tune(kv, neg):
    cands = sorted({r['P_cands'][0]['sim'] + 1e-6 for r in neg}) + [-1.0]
    ok = [c for c in cands if neg_wrong(neg, c) <= 0.10]
    t = min(ok) if ok else max(cands)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['tune', 'score'])
    ap.add_argument('--kv', nargs='+', required=True)
    ap.add_argument('--neg', nargs='*', default=[])
    ap.add_argument('--t', type=float)
    a = ap.parse_args()
    kv, neg = load(a.kv), load(a.neg)
    if a.mode == 'tune':
        t = tune(kv, neg)
        print(json.dumps(dict(t=round(t, 6), tune=acc(kv, t), tune_neg_wrong_fetch=neg_wrong(neg, t))))
    else:
        out = dict(t=a.t, **acc(kv, a.t))
        if neg:
            out['neg_wrong_fetch'] = neg_wrong(neg, a.t)
        print(json.dumps(out))


if __name__ == '__main__':
    main()
