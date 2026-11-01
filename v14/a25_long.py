"""A25 (registered 2026-11-01 15:03 IST): the PACKAGE at 128k / 300k, by needle DISTANCE.

Docs from bench's parts (held-out prose, bench keys / values / templates): 8 facts per doc; the ASKED facts sit at fixed
distances from the question (one per bin: ~1k, ~16k, ~100k, ~250k at 300k); each asked fact is a separate question on
the SAME streamed session. Reads: package lenient / strict (v16/package.py), the package's F2 in the sign-bit form,
mind alone (session state + question), T++ 2k window alone, T++ 2k window + the package's row + contrast.

    python v14/a25_long.py --lengths 131072,307200 --out logs/v14_a25_long_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from v16.bench import Reader, natural_key  # noqa: E402
from v7.data import _RECALL_VALUE_NOUNS, _recall_key  # noqa: E402
from v14.a6_reread import sent_ok  # noqa: E402
from v14.a18_tpp_warm import TPP, tpp_greedy  # noqa: E402
from v14.styles import QUESTIONS  # noqa: E402
from v16.package import Package  # noqa: E402

BINS = {131072: [1000, 16000, 100000], 307200: [1000, 16000, 100000, 250000]}
CELLS = ('kv', 'kvV', 'para', 'tool', 'neg')
FIELDS = ('status', 'owner', 'region', 'label')
WINDOW = 2048


def make_doc(rng, tok, fill, L, cell, keys):
    """(ids, questions): questions = [(bin distance, question text, value or None for never-stored)]."""
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    start = rng.randrange(0, len(fill) - L - 1)
    body = fill[start:start + L].tolist()
    mk = (lambda: natural_key(rng, tok, fill, start, start + L)) if keys == 'natural' else (lambda: _recall_key(rng))
    bins = BINS.get(L, [min(1000, L // 4), L // 2])                     # (smoke lengths)
    facts, asked = [], []
    for i in range(8):
        k = mk(); v = rng.choice(_RECALL_VALUE_NOUNS)
        if cell == 'tool':
            vals = [rng.choice(_RECALL_VALUE_NOUNS) for _ in FIELDS]
            text = f" Result: job: {k}; " + "; ".join(f"{f}: {x}" for f, x in zip(FIELDS, vals)) + "."
            fi = rng.randrange(len(FIELDS)); v = vals[fi]; q = f" Query: job: {k}; {FIELDS[fi]}:"
        else:
            text = f" Definition: {k} refers to {v}."
            if cell == 'kvV':
                q = rng.choice(QUESTIONS[1:])[1].format(k=k)               # a natural (non-template) wording
            elif cell == 'para':
                w = k.split(); last = w[-1]
                q = f" Query: {rng.choice([k.title(), k.upper(), ' '.join(w[:-1] + [last[:-1] if last.endswith('s') else last + 's']), 'the ' + k])} refers to"
            else:
                q = f" Query: {k} refers to"
        facts.append((enc(text), q, v))
    # positions: the first len(bins) facts at the bin distances from the end, the rest at random positions
    pos = [L - d for d in bins] + [rng.randrange(0, L - 64) for _ in range(8 - len(bins))]
    order = sorted(range(8), key=lambda i: pos[i])
    out, cur = [], 0
    for i in order:
        p = pos[i]
        out += body[cur:p]; out += facts[i][0]; cur = p
    out += body[cur:]
    if cell == 'neg':                                                    # never-stored keys, one question per bin
        asked = [(d, f" Query: {mk()} refers to", None) for d in bins]
    else:
        asked = [(d, facts[i][1], facts[i][2]) for i, d in enumerate(bins)]
    return out, asked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lengths', default='131072,307200'); ap.add_argument('--docs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=18500); ap.add_argument('--device', default='cuda')
    ap.add_argument('--cells', default=','.join(CELLS)); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pkg = Package(device=a.device); tok = pkg.tok
    tm, _, _ = _load_v13_sempty(Path(TPP), 'baseline_real_pm', torch.device(a.device)); tm.eval()
    T = Reader(tm, 'transformer')
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    fill = filler_tokens(str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    outf = open(a.out, 'a')
    for L in [int(x) for x in a.lengths.split(',')]:
        for cell in a.cells.split(','):
            for keys in ('synthetic', 'natural'):
                rng = random.Random(a.seed * 1000 + L + CELLS.index(cell) * 101 + (keys == 'natural'))
                for d in range(a.docs):
                    ids, asked = make_doc(rng, tok, fill, L, cell, keys)
                    t0 = time.time(); pkg.read(ids); t_read = time.time() - t0
                    win = ids[-WINDOW:]
                    for dist, q, v in asked:
                        qi = enc(q)
                        t1 = time.time()
                        ans, info = pkg.answer(ids, q, mode='lenient')
                        t_ans = time.time() - t1
                        rec = dict(L=L, cell=cell, keys=keys, doc=d, dist=dist, negative=int(v is None), path=info['path'],
                                   read_sec=round(t_read, 1), answer_sec=round(t_ans, 2), bytes_held=info['bytes_held'],
                                   question=q, package=ans, row=(info.get('row_text') or '')[-90:])
                        ok = lambda t: (0 if v is None else sent_ok(t, v))  # noqa: E731
                        rec['package_ok'] = ok(ans)
                        rec['strict_ok'] = ok(ans) if info['path'] != 'F2' else 0
                        rec['strict_abstain'] = int(info['path'] in (None, 'F2'))
                        if info['path'] == 'F2':
                            a2, _ = pkg.answer(ids, q, mode='lenient', address='sal1'); rec['package_sal1_ok'] = ok(a2)
                        # mind alone: the session state + question, plain greedy (no row, no contrast)
                        s = pkg._session; st = copy.deepcopy(s['state'])
                        la, st, off = pkg.mind.feed(qi, st, s['off']); out = []
                        for _ in range(16):
                            t = int(la.argmax()); out.append(t)
                            if t in pkg.stops:
                                break
                            la, st, off = pkg.mind.feed([t], st, off)
                        rec['mind_alone'] = tok.decode(out); rec['mind_alone_ok'] = ok(rec['mind_alone'])
                        g = tpp_greedy(T, win + qi); rec['tpp_window'] = tok.decode(g); rec['tpp_window_ok'] = ok(rec['tpp_window'])
                        if info['row'] is not None:
                            r0, r1 = info['row']; row = ids[r0:r1]
                            g = tpp_greedy(T, win + row + qi, win + qi); rec['tpp_window_row_ok'] = ok(tok.decode(g))
                        outf.write(json.dumps(rec) + '\n'); outf.flush()
                    print(L, cell, keys, d, 'read', round(t_read, 1), 's', flush=True)


if __name__ == '__main__':
    main()
