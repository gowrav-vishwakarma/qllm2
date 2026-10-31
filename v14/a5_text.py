"""A5 text part (pre-registered 2026-11-01 01:13 IST): on the SYNONYM cell, can text find the fact?
Organ chain (token organ + A4 normalised organ), BM25 over the doc's sentences, MiniLM-L6 over the doc's
sentences; each scored by whether its top-1 sentence holds the asked fact ("{A} refers to {v}").
CPU only (MiniLM capped at --threads).

    python v14/a5_text.py --seed 25000 --L 32768 --docs 40 --out logs/v14_a5_text_<hash>_<stamp>.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from memory_probes.spread import filler_tokens  # noqa: E402
from v7.data import get_chat_tokenizer  # noqa: E402
from v14 import norm_organ as N  # noqa: E402
from v14 import package_fetch as P  # noqa: E402
from v14.shared_fetch import BM25, minilm_embed, norm_words  # noqa: E402
from v14.syn_cell import build_syn  # noqa: E402

FILL = ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'
SENT = re.compile(r'(?<=[.!?])\s+')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=25000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=40); ap.add_argument('--needles', type=int, default=8)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--fill', default=str(FILL)); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    tok = get_chat_tokenizer(); P._S['strs'] = N.tok_strings(tok)
    fill = filler_tokens(a.fill)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    res = {}
    for task in ('syn', 'synneg'):
        rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(task.encode()) % 997)   # = bench.py main()
        c = dict(n=0, organ_hit=0, organ_right=0, bm25_right=0, minilm_right=0, bm25_def=0, minilm_def=0)
        ex = []
        t0 = time.time()
        for d in range(a.docs):
            ids, ans, qlen, meta = build_syn(rng, tok, filler, a.L, a.needles, neg=(task == 'synneg'))
            T = len(ids); end = T - qlen
            fact = f"{meta['key']} refers to {meta['val']}"
            hit = P.organ(ids, end, T)
            row = P.row_around(ids, hit, hit, end) if hit is not None else None
            if row is None:
                nh = P.norm_hit(ids, end, end)
                row = nh[2:] if nh else None
            c['n'] += 1; c['organ_hit'] += int(row is not None)
            c['organ_right'] += int(row is not None and fact in tok.decode(ids[row[0]:row[1]]))
            sents = [s for s in SENT.split(tok.decode(ids[:end])) if s.strip()]
            qtext = tok.decode(ids[end:]).strip()
            bm = BM25([norm_words(s) for s in sents]).rank(norm_words(qtext), k=1)
            b1 = sents[bm[0]] if bm else ''
            E = minilm_embed(sents + [qtext])
            m1 = sents[int(np.argmax(E[:-1] @ E[-1]))]
            c['bm25_right'] += int(fact in b1); c['minilm_right'] += int(fact in m1)
            c['bm25_def'] += int('Definition:' in b1)
            c['minilm_def'] += int('Definition:' in m1)
            if len(ex) < 4:
                ex.append(dict(q=qtext, fact=fact, bm25=b1[-120:], minilm=m1[-120:]))
            print(f"{task} doc {d} n_sent {len(sents)} organ {int(row is not None)} bm25 {int(fact in b1)} "
                  f"minilm {int(fact in m1)}  {time.time() - t0:.0f}s", flush=True)
        r = {k: (round(v / c['n'], 4) if k != 'n' else v) for k, v in c.items()}
        res[task] = dict(r, examples=ex)
        print(task, json.dumps(r), flush=True)
    json.dump(dict(args=vars(a), cells=res), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
