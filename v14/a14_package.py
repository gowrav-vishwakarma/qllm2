"""A14 (pre-registered 2026-11-01 06:02 IST): the 3-step lenient package with its memory stated.
A15 (06:59 IST): --forms all8 / sal8 / sal1 keep the full 588-d direction at int8 / sign bits.

Per doc (bench cells): (1) organ chain (token organ, then the normalised organ on the whole question); (2) if it
misses, the surprise-keyed address (A13's F2: the 2 most surprising question tokens, block-8 max-sim over each
sentence's tokens) in three stored forms: V-full (every token, 588-d), V-pca (every token, shipped PCA-64, int8),
V-sal (only the stream's top-1/16 surprising tokens, PCA-64 int8); (3) read warm + no-row contrast. Strict =
organ only, abstain otherwise.

    python v14/a14_package.py --seed 34000 --docs 30 --out logs/v14_a14_package_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from memory_probes.spread import filler_tokens  # noqa: E402
from v16.bench import Reader, build  # noqa: E402
from v7.data import _RECALL_VALUE_NOUNS  # noqa: E402
from v14 import package_fetch as P  # noqa: E402
from v14.a6_reread import CKPT, FILL, Mind, sent_ok  # noqa: E402
from v14.a13_manykey import capture, sentence_rows  # noqa: E402

PCA = '/media/gowrav/d_drive/qllm_ckpt/v14/a14_pca64_b2p2.pt'
CELLS = ('kv/synthetic,para/synthetic,tool/synthetic,neg/synthetic,kv/natural,para/natural,tool/natural,neg/natural,'
         'syn/natural,synneg/natural,natq/natural,natqneg/natural')
ALPHA, SAL_FRAC = 4.0, 1 / 16
VALUE_RE = re.compile(r'\b(' + '|'.join(_RECALL_VALUE_NOUNS) + r')\b', re.I)


def int8(x):
    """per-vector symmetric int8 quantisation, returned dequantised (what a reader of the stored form sees)."""
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(x / s).clamp(-127, 127) * s


def rank_rows(V, keys, rows, keep=None):
    """sum over keys of the max cosine over a row's (kept) tokens; rows with no kept token score -inf."""
    sims = V @ keys.T
    out = []
    for a, b in rows:
        if keep is None:
            out.append(sims[a:b].max(0).values.sum())
        else:
            m = keep[a:b]
            out.append(sims[a:b][m].max(0).values.sum() if bool(m.any()) else torch.tensor(-1e9, device=V.device))
    return torch.stack(out).argsort(descending=True).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=34000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=30); ap.add_argument('--needles', type=int, default=8)
    ap.add_argument('--cells', default=CELLS); ap.add_argument('--device', default='cuda')
    ap.add_argument('--forms', default='full,pca,sal', help='A15 adds all8 / sal8 / sal1 (full 588-d at int8 / sign bits)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    mind = Mind(CKPT, dev); tok = mind.tok; R = Reader(mind.m, 'pam')
    B = torch.load(PCA, map_location='cpu')['basis'].float().to(dev)          # [588, 64]
    fill = filler_tokens(FILL)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    outf = open(a.out, 'a')
    for cell in a.cells.split(','):
        task, keys = cell.split('/')
        rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(task.encode()) % 997)   # = bench.py main()
        for d in range(a.docs):
            t0 = time.time()
            ids, ans, qlen, meta = build(rng, tok, filler, a.L, task, a.needles, keys=keys)
            end = len(ids) - qlen; q = ids[end:]
            neg = bool(meta.get('negative')); val = str(meta['val'])
            rec = dict(task=task, keys=keys, seed=a.seed, doc=d, L=a.L, negative=int(neg))
            # step 1: organ chain (text)
            hit = P.organ(ids, end, len(ids))
            orow = P.row_around(ids, hit, hit, end) if hit is not None else None
            if orow is None:
                nh = P.norm_hit(ids, end, end)
                orow = nh[2:] if nh else None
            rec['organ_hit'] = int(orow is not None)
            # step 2: the surprise-keyed address in three stored forms
            rows = sentence_rows(ids, end)
            texts = [tok.decode(ids[x:y]) for x, y in rows]
            key = meta['key']
            holds = lambda s: (not neg) and key in s and re.search(r'\b' + re.escape(val) + r'\b', s) is not None
            fact = [i for i, s in enumerate(texts) if holds(s)]
            got, pre = capture(R, ids, (8,))
            surp = R.surprisal(ids, pre.cpu())
            qs = surp[end:].clone(); qs[:2] = -1.0
            sal = [end + int(i) for i in qs.topk(2).indices]
            v = got[8]; c = v - v[:end].mean(0)
            full = torch.nn.functional.normalize(c, dim=-1)
            pca = torch.nn.functional.normalize(int8(c @ B), dim=-1)                    # stored form: int8
            qp = torch.nn.functional.normalize(c[sal] @ B, dim=-1)                      # query: projected, not stored
            k = max(1, int(round(end * SAL_FRAC)))
            keep = torch.zeros(len(ids), dtype=torch.bool, device=dev)
            keep[surp[:end].topk(k).indices.to(dev)] = True
            want = a.forms.split(',')
            q8 = torch.nn.functional.normalize(c[sal], dim=-1)                          # query: full precision
            make = {'full': lambda: rank_rows(full[:end], full[sal], rows),
                    'pca': lambda: rank_rows(pca[:end], qp, rows),
                    'sal': lambda: rank_rows(pca[:end], qp, rows, keep[:end]),
                    'all8': lambda: rank_rows(torch.nn.functional.normalize(int8(c[:end]), dim=-1), q8, rows),
                    'sal8': lambda: rank_rows(torch.nn.functional.normalize(int8(c[:end]), dim=-1), q8, rows, keep[:end]),
                    'sal1': lambda: rank_rows(torch.nn.functional.normalize(torch.sign(c[:end]), dim=-1),
                                              torch.nn.functional.normalize(torch.sign(c[sal]), dim=-1), rows, keep[:end])}
            ranks = {name: make[name]() for name in want}
            for name, rk in ranks.items():
                pos = [rk.index(i) for i in fact] if fact else []
                rec[f'{name}_rank'] = min(pos) if pos else None
            rec['n_salient'] = k; rec['doc_tokens'] = end
            # step 3: read warm + contrast (organ row if hit, else each form's top-1)
            _, st, off = mind.feed(ids[:end])
            cache = {}
            def warmc(span):
                key_ = tuple(span)
                if key_ not in cache:
                    g = mind.greedy_contrast(list(span) + q, copy.deepcopy(st), off, q, copy.deepcopy(st), off, ALPHA)
                    cache[key_] = tok.decode(g)
                return cache[key_]
            if orow is not None:
                txt = warmc(ids[orow[0]:orow[1]])
                rec['strict_text'] = txt; rec['strict'] = sent_ok(txt, val) if not neg else 0
                rec['strict_value_word'] = int(VALUE_RE.search(txt) is not None)
                for name in ranks:
                    rec[f'lenient_{name}'] = rec['strict']; rec[f'lenient_{name}_value_word'] = rec['strict_value_word']
            else:
                rec['strict'] = 0; rec['strict_value_word'] = 0
                for name, rk in ranks.items():
                    r0, r1 = rows[rk[0]]
                    txt = warmc(ids[r0:r1])
                    rec[f'lenient_{name}_text'] = txt
                    rec[f'lenient_{name}'] = sent_ok(txt, val) if not neg else 0
                    rec[f'lenient_{name}_value_word'] = int(VALUE_RE.search(txt) is not None)
            rec['sec'] = round(time.time() - t0, 1)
            outf.write(json.dumps(rec) + '\n'); outf.flush()
            print(cell, d, 'organ', rec['organ_hit'], {n: rec[f'{n}_rank'] for n in ranks},
                  {n: rec[f'lenient_{n}'] for n in ranks}, rec['strict'], rec['sec'], flush=True)


if __name__ == '__main__':
    main()
