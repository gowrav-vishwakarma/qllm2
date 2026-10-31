"""A12 (pre-registered 2026-11-01 05:09 IST): many-key TEXT fetch on natq / natqneg (CPU, no mind).

R1 = BM25 over the doc's sentences with the whole question; R2 = the rarest contiguous question span (3, 2, 1
normalised words) present in the stream, row = the sentence of its latest occurrence, abstain if its count > 2.
Docs come from v16/bench.py build() with the harness's rng, so they are exactly bench's.

    python v14/a12_natq_text.py --seed 32000 --docs 40 --out logs/v14_a12_natq_text_<hash>_<stamp>.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from memory_probes.spread import filler_tokens  # noqa: E402
from v16.bench import build  # noqa: E402
from v7.data import get_chat_tokenizer  # noqa: E402
from v14.shared_fetch import BM25, norm_words  # noqa: E402

FILL = '/home/gowrav/Development/qllm2/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'
SENT = re.compile(r'(?<=[.!?])\s+')


def auroc(pos, neg):
    if not pos or not neg:
        return float('nan')
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def rarest_span(sents_w, qw):
    """(sentence index, span, count) of the rarest question span present in the stream, or None."""
    stream = [w for s in sents_w for w in s]
    owner = [i for i, s in enumerate(sents_w) for _ in s]
    best = None
    for n in (3, 2, 1):
        for i in range(len(qw) - n + 1):
            g = tuple(qw[i:i + n]); cnt = 0; last = None
            for j in range(len(stream) - n + 1):
                if tuple(stream[j:j + n]) == g:
                    cnt += 1; last = j
            if cnt and (best is None or cnt < best[2] or (cnt == best[2] and n > len(best[1]))):
                best = (owner[last], g, cnt)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=32000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=40); ap.add_argument('--needles', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tok = get_chat_tokenizer()
    fill = filler_tokens(FILL)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    res, scores = {}, {}
    for keys in ('synthetic', 'natural'):
        for task in ('natq', 'natqneg'):
            rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(task.encode()) % 997)   # = bench.py main()
            c = Counter(); top1 = []; ex = []
            for d in range(a.docs):
                ids, ans, qlen, meta = build(rng, tok, filler, a.L, task, a.needles, keys=keys)
                end = len(ids) - qlen
                sents = [s for s in SENT.split(tok.decode(ids[:end])) if s.strip()]
                sw = [norm_words(s) for s in sents]
                q = tok.decode(ids[end:]); qw = norm_words(q)
                key, val = meta['key'], str(meta['val'])
                fact = lambda s: key in s and re.search(r'\b' + re.escape(val) + r'\b', s) is not None
                bm = BM25(sw); r1 = bm.rank(qw, k=5)
                c['n'] += 1
                c['r1_top1'] += int(bool(r1) and fact(sents[r1[0]]))
                c['r1_top5'] += int(any(fact(sents[i]) for i in r1))
                rs = rarest_span(sw, qw)
                abstain = rs is None or rs[2] > 2
                c['r2_abstain'] += int(abstain)
                c['r2_right'] += int(not abstain and fact(sents[rs[0]]))
                if len(ex) < 3:
                    ex.append(dict(q=q, key=key, span=' '.join(rs[1]) if rs else None, count=rs[2] if rs else 0,
                                   r2_row=sents[rs[0]][-100:] if rs else None))
                top1.append(_top1_score(bm, qw))                         # R1's best score: natq vs natqneg AUROC
            n = c['n']
            res[f'{task}/{keys}'] = dict({k: (round(v / n, 4) if k != 'n' else v) for k, v in c.items()}, examples=ex)
            scores[(task, keys)] = top1
            print(task, keys, json.dumps({k: round(v / n, 3) for k, v in c.items() if k != 'n'}), flush=True)
            for e in ex[:2]:
                print('   ', json.dumps(e)[:220], flush=True)
    for keys in ('synthetic', 'natural'):
        res[f'r1_score_auroc/{keys}'] = round(auroc(scores[('natq', keys)], scores[('natqneg', keys)]), 4)
        print('R1 top-1 score AUROC natq vs natqneg', keys, res[f'r1_score_auroc/{keys}'])
    ok = all(res[f'natq/{k}']['r2_right'] >= 0.90 and res[f'natqneg/{k}']['r2_abstain'] >= 0.90 for k in ('synthetic', 'natural'))
    res['bar'] = 'MET' if ok else 'NOT met'
    print('BAR (R2 natq right >= 0.90 and natqneg abstain >= 0.90, both key types):', res['bar'])
    json.dump(dict(args=vars(a), cells=res), open(a.out, 'w'), indent=1)


def _top1_score(bm, qw):
    """the best BM25 score over the doc's sentences (BM25.rank only returns indices)."""
    import math
    best = 0.0
    for i, tf in enumerate(bm.tf):
        s = 0.0
        for w in set(qw):
            if w in tf and w in bm.idf:
                f = tf[w]
                s += bm.idf[w] * f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * bm.len[i] / bm.avg))
        best = max(best, s)
    return best


if __name__ == '__main__':
    main()
