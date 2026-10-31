"""A11 (pre-registered 2026-11-01 05:04 IST, amended 05:05): key-swap relevance.

rel = log p(v | row, q) - mean_k log p(v | row, q_swap_k): v = the mind's greedy first answer token given row + q;
q_swap_k = the question with its key replaced by a never-stored natural phrase. A right row's answer should depend
on the key, a wrong row's should not. Cold short reads (row + question, fresh state), no training, CPU.
Rows per doc (bench kv docs, natural keys): the asked key's fact (right), another stored key's fact in the same
template (hard wrong), a prose sentence (easy wrong).

    python v14/a11_keyswap.py --seed 31000 --docs 40 --out logs/v14_a11_keyswap_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from memory_probes.spread import filler_tokens  # noqa: E402
from v16.bench import build, natural_key  # noqa: E402
from v7.data import _RECALL_VALUE_NOUNS  # noqa: E402
from v14 import package_fetch as P  # noqa: E402
from v14.a6_reread import CKPT, FILL, Mind, fact_row  # noqa: E402

CU1 = '/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/cu1_best_2efb422.pt'
K_SWAP = 3


def occurrences(ids, pat, end):
    return [i for i in range(end - len(pat) + 1) if ids[i] == pat[0] and ids[i:i + len(pat)] == pat]


def rows_for(ids, end, tok, meta, rng):
    """[(kind, (a, b))]: the right fact, another stored fact in the same template, a prose sentence."""
    out = []
    right = fact_row(ids, end, tok, meta)
    if right is None:
        return out
    out.append(('right', right))
    defs = []
    for pos in occurrences(ids, tok.encode(" Definition:", add_special_tokens=False), end):
        r = P.row_around(ids, pos, pos + 3, end)
        if meta['key'] not in tok.decode(ids[r[0]:r[1]]):
            defs.append(r)
    if defs:
        out.append(('def', rng.choice(defs)))
    for _ in range(100):
        p = rng.randrange(64, end - 64)
        r = P.row_around(ids, p, p, end)
        txt = tok.decode(ids[r[0]:r[1]])
        if r[1] - r[0] >= 8 and 'Definition' not in txt and 'Query' not in txt:
            out.append(('prose', r))
            break
    return out


def swap_keys(rng, tok, filler, doc_text, k):
    keys = []
    while len(keys) < k:
        c = natural_key(rng, tok, filler, 0, 0)
        if c not in doc_text and c not in keys:
            keys.append(c)
    return keys


@torch.no_grad()
def read(mind, row, q, swaps):
    lg, _, _ = mind.feed(row + q)
    lp = torch.log_softmax(lg.float(), -1)
    v = int(lp.argmax())
    lp1 = float(lp[v])
    lp2 = []
    for qs in swaps:
        lg2, _, _ = mind.feed(row + qs)
        lp2.append(float(torch.log_softmax(lg2.float(), -1)[v]))
    return v, lp1, float(np.mean(lp2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=31000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=40); ap.add_argument('--needles', type=int, default=8)
    ap.add_argument('--minds', default='b2p2,cu1'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    paths = {'b2p2': CKPT, 'cu1': CU1}
    minds = {n: Mind(paths[n], dev) for n in a.minds.split(',')}
    tok = next(iter(minds.values())).tok
    cand = {tok.encode(' ' + v, add_special_tokens=False)[0] for v in _RECALL_VALUE_NOUNS}
    fill = filler_tokens(FILL)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(b'kv') % 997)          # = bench.py main() for task kv
    outf = open(a.out, 'a')
    for d in range(a.docs):
        t0 = time.time()
        ids, ans, qlen, meta = build(rng, tok, filler, a.L, 'kv', a.needles, keys='natural')
        end = len(ids) - qlen; q = ids[end:]
        side = random.Random(a.seed * 7919 + d)                                  # row / swap sampling, apart from bench's rng
        rows = rows_for(ids, end, tok, meta, side)
        swaps = [tok.encode(f" Query: {k} refers to", add_special_tokens=False)
                 for k in swap_keys(side, tok, filler, tok.decode(ids[:end]), K_SWAP)]
        for kind, (r0, r1) in rows:
            row = ids[r0:r1]
            for name, mind in minds.items():
                v, lp1, lp2 = read(mind, row, q, swaps)
                outf.write(json.dumps(dict(doc=d, seed=a.seed, kind=kind, mind=name, rel=lp1 - lp2, conf=lp1,
                                           lp_swap=lp2, v=v, v_text=tok.decode([v]), v_is_value=int(v in cand),
                                           v_correct=int(v == ans), row_text=tok.decode(row)[-120:])) + '\n')
        outf.flush()
        print(d, [k for k, _ in rows], round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
