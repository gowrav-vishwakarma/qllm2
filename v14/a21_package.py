"""A21 (pre-registered 2026-11-01 11:02 IST): the question-time package on NATURAL variants (varied question wording).

Fetch: token organ -> word organ steps 1-2 -> step 4 (A20) [text_fetch(step4=True, bigram=False)] -> F2 (A13: the 2 most
surprising question tokens, block-8 max-sim over the doc's sentence rows, full-precision states). Read: b2p2 warm +
no-row contrast (A9). STRICT twin = text steps only (abstain on a miss).

    python v14/a21_package.py --out logs/v14_a21_package_<hash>_<stamp>.jsonl
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
from v14.a6_reread import CKPT, FILL, Mind, sent_ok  # noqa: E402
from v14.a13_manykey import capture, maxsim_scores, sentence_rows  # noqa: E402
from v14.package_fetch import text_fetch  # noqa: E402

CELLS = ('kv/synthetic,kv/natural,kv1/synthetic,kv1/natural,kvupd/synthetic,kvupd/natural,natq/natural,'
         'neg/synthetic,neg/natural,para/synthetic,para/natural,tool/synthetic,tool/natural')
CONTROLS = ('para', 'tool')
ALPHA = 4.0
VALUE_RE = re.compile(r'\b(' + '|'.join(_RECALL_VALUE_NOUNS) + r')\b', re.I)
FIELDS = ('status', 'owner', 'region', 'label')


def holds(text, task, meta, q):
    key, val = meta['key'], str(meta['val'])
    if meta.get('negative'):
        return False
    if task == 'tool':
        f = next((x for x in FIELDS if x in q), None)
        return f"job: {key};" in text and f is not None and f"{f}: {val}" in text
    return key in text and re.search(r'\b' + re.escape(val) + r'\b', text) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=18000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=40); ap.add_argument('--control_docs', type=int, default=20)
    ap.add_argument('--cells', default=CELLS); ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    mind = Mind(CKPT, dev); tok = mind.tok; R = Reader(mind.m, 'pam')
    fill = filler_tokens(FILL)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    outf = open(a.out, 'a')
    for cell in a.cells.split(','):
        task, keys = cell.split('/')
        n_docs = a.control_docs if task in CONTROLS else a.docs
        rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(task.encode()) % 997)   # = bench.py main()
        for d in range(n_docs):
            t0 = time.time()
            ids, ans, qlen, meta = build(rng, tok, filler, a.L, task, 8, keys=keys, qstyle='varied')
            end = len(ids) - qlen; q = ids[end:]; qt = tok.decode(q); val = str(meta['val'])
            rec = dict(task=task, keys=keys, seed=a.seed, doc=d, L=a.L, negative=int(bool(meta.get('negative'))), query_text=qt)
            trow, info = text_fetch(ids, end, step4=True, bigram=False)
            rec['text_step'] = info['step']
            rec['text_right'] = int(trow is not None and holds(tok.decode(ids[trow[0]:trow[1]]), task, meta, qt))
            row, path = trow, (f"text{info['step']}" if trow is not None else None)
            if row is None:                                       # F2: the mind's surprise-keyed address
                rows = sentence_rows(ids, end)
                got, pre = capture(R, ids, (8,))
                surp = R.surprisal(ids, pre.cpu())
                qs = surp[end:].clone(); qs[:2] = -1.0
                sal = [end + int(i) for i in qs.topk(2).indices]
                v = got[8]; V = torch.nn.functional.normalize(v - v[:end].mean(0), dim=-1)
                sc = maxsim_scores(V[:end], V[sal], rows)
                row, path = rows[int(sc.argmax())], 'F2'
                rec['salient'] = [tok.decode([ids[p]]) for p in sal]
            rec['path'] = path
            rec['chain_right'] = int(holds(tok.decode(ids[row[0]:row[1]]), task, meta, qt))
            _, st, off = mind.feed(ids[:end])
            g = mind.greedy_contrast(ids[row[0]:row[1]] + q, copy.deepcopy(st), off, q, copy.deepcopy(st), off, ALPHA)
            txt = tok.decode(g)
            rec['answer_text'] = txt
            rec['lenient'] = 0 if rec['negative'] else sent_ok(txt, val)
            rec['lenient_value_word'] = int(VALUE_RE.search(txt) is not None)
            rec['strict'] = rec['lenient'] if trow is not None else 0          # strict: text steps only
            rec['strict_value_word'] = rec['lenient_value_word'] if trow is not None else 0
            rec['sec'] = round(time.time() - t0, 1)
            outf.write(json.dumps(rec) + '\n'); outf.flush()
            print(cell, d, path, rec['text_right'], rec['chain_right'], rec['lenient'], rec['sec'], flush=True)


if __name__ == '__main__':
    main()
