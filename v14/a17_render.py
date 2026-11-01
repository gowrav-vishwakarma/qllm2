"""A17 (pre-registered 2026-11-01 08:47 IST): render the fetched row + question in the mind's TRAINED template.

Rows come from the strict organ chain (text). The library parses what it stored (bench's tool / definition rows)
and re-renders it in cu2's fld format: " Record: K | f: v | ...." and " Q: What is the f of K? A:". Each answering
mind (cu2a, b2p2) reads RAW and RENDERED, warm + contrast (its own doc state) and cold.

    python v14/a17_render.py --seed 36000 --docs 30 --out logs/v14_a17_render_<hash>_<stamp>.jsonl
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
from v16.bench import build  # noqa: E402
from v14 import package_fetch as P  # noqa: E402
from v14.a6_reread import CKPT, FILL, Mind, sent_ok  # noqa: E402

CU2A = '/home/gowrav/Development/qllm2/checkpoints_v13_sempty/cu2a_main_24b2969/best_model.pt'
CELLS = 'tool/synthetic,tool/natural,kv/synthetic,kv/natural,para/synthetic,para/natural'
ALPHA = 4.0
TOOL_ROW = re.compile(r'Result: job: (.+?); (.+?)\.')
TOOL_Q = re.compile(r'Query: job: (.+?); (\w+):')
DEF_ROW = re.compile(r'Definition: (.+?) refers to (\w+)\.')


def render(task, row_text, q_text):
    """(rendered row text, rendered question text) or None when the row / question does not parse."""
    if task == 'tool':
        r, q = TOOL_ROW.search(row_text), TOOL_Q.search(q_text)
        if not r or not q:
            return None
        fields = [f.split(': ', 1) for f in r.group(2).split('; ') if ': ' in f]
        return (f" Record: {r.group(1)} | " + ' | '.join(f'{f}: {v}' for f, v in fields) + '.',
                f" Q: What is the {q.group(2)} of {r.group(1)}? A:")
    r = DEF_ROW.search(row_text)
    if not r:
        return None
    return f" Record: {r.group(1)} | meaning: {r.group(2)}.", f" Q: What is the meaning of {r.group(1)}? A:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=36000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=30); ap.add_argument('--needles', type=int, default=8)
    ap.add_argument('--cells', default=CELLS); ap.add_argument('--minds', default='cu2a,b2p2')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    paths = {'cu2a': CU2A, 'b2p2': CKPT}
    minds = {n: Mind(paths[n], dev) for n in a.minds.split(',')}
    tok = next(iter(minds.values())).tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    fill = filler_tokens(FILL)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    outf = open(a.out, 'a')
    for cell in a.cells.split(','):
        task, keys = cell.split('/')
        rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(task.encode()) % 997)   # = bench.py main()
        for d in range(a.docs):
            t0 = time.time()
            ids, ans, qlen, meta = build(rng, tok, filler, a.L, task, a.needles, keys=keys)
            end = len(ids) - qlen; q = ids[end:]; val = str(meta['val'])
            hit = P.organ(ids, end, len(ids))
            row = P.row_around(ids, hit, hit, end) if hit is not None else None
            if row is None:
                nh = P.norm_hit(ids, end, end)
                row = nh[2:] if nh else None
            rec = dict(task=task, keys=keys, seed=a.seed, doc=d, L=a.L, organ_hit=int(row is not None))
            if row is not None:
                span = ids[row[0]:row[1]]
                rr = render(task, tok.decode(span), tok.decode(q))
                rec['parse_ok'] = int(rr is not None)
                rspan, rq = (enc(rr[0]), enc(rr[1])) if rr else (span, q)
                if rr:
                    rec['rendered'] = rr[0][-90:] + ' ||' + rr[1]
                for name, mind in minds.items():
                    _, st, off = mind.feed(ids[:end])
                    for arm, sp, qq in (('raw', span, q), ('ren', rspan, rq)):
                        g = mind.greedy_contrast(sp + qq, copy.deepcopy(st), off, qq, copy.deepcopy(st), off, ALPHA)
                        txt = tok.decode(g); rec[f'{name}_{arm}_warmc'] = sent_ok(txt, val); rec[f'{name}_{arm}_warmc_text'] = txt
                        txt = tok.decode(mind.greedy(sp + qq)); rec[f'{name}_{arm}_cold'] = sent_ok(txt, val); rec[f'{name}_{arm}_cold_text'] = txt
            rec['sec'] = round(time.time() - t0, 1)
            outf.write(json.dumps(rec) + '\n'); outf.flush()
            print(cell, d, rec['organ_hit'], rec.get('parse_ok'), {k: rec[k] for k in rec if k.endswith('_warmc')}, rec['sec'], flush=True)


if __name__ == '__main__':
    main()
