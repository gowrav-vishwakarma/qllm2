"""A4 text-only check (pre-registered 2026-11-01 01:04 IST): does the organ chain (token organ, then the
normalised-word organ) find the RIGHT row on bench's own documents, and stay silent on never-stored keys?
CPU only: the documents come from v16/bench.py's build() with the harness's own rng, so they are exactly
the docs a bench run with the same seed / length / task would see.

    python v14/a4_check.py --seed 24000 --L 32768 --docs 40 --out logs/v14_a4_text_<hash>_<stamp>.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from v16.bench import build  # noqa: E402
from memory_probes.spread import filler_tokens  # noqa: E402
from v7.data import get_chat_tokenizer  # noqa: E402
from v14 import package_fetch as P  # noqa: E402
from v14 import norm_organ as N  # noqa: E402

FILL = ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'


def right_row(text, task, meta, qtext):
    if task in ('kv', 'para'):
        return f"{meta['key']} refers to {meta['val']}" in text
    if task == 'tool':
        field = qtext.rstrip().rstrip(':').split()[-1]
        return f"job: {meta['key']};" in text and f"{field}: {meta['val']}" in text
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=24000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=40); ap.add_argument('--needles', type=int, default=8)
    ap.add_argument('--fill', default=str(FILL)); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tok = get_chat_tokenizer(); P._S['strs'] = N.tok_strings(tok)
    fill = filler_tokens(a.fill)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    res = {}
    for keys in ('synthetic', 'natural'):
        for task in ('kv', 'para', 'tool', 'neg'):
            rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(task.encode()) % 997)   # = bench.py main()
            c = dict(n=0, token_hit=0, norm_hit=0, step1=0, step2=0, step3=0, any_hit=0, right=0, sec=0.0)
            miss = []
            for d in range(a.docs):
                t0 = time.time()
                ids, ans, qlen, meta = build(rng, tok, filler, a.L, task, a.needles, keys=keys)
                T = len(ids); end = T - qlen
                hit = P.organ(ids, end, T)
                row = P.row_around(ids, hit, hit, end) if hit is not None else None
                c['n'] += 1; c['token_hit'] += int(row is not None)
                if row is None:
                    nh = P.norm_hit(ids, end, end)
                    if nh:
                        c['norm_hit'] += 1; c[f'step{nh[0]}'] += 1; row = nh[2:]
                c['any_hit'] += int(row is not None)
                qtext = tok.decode(ids[end:])
                ok = row is not None and right_row(tok.decode(ids[row[0]:row[1]]), task, meta, qtext)
                c['right'] += int(ok)
                if (task == 'neg' and row is not None) or (task != 'neg' and not ok):
                    miss.append(dict(doc=d, q=qtext, row=tok.decode(ids[row[0]:row[1]]) if row else None))
                c['sec'] += time.time() - t0
            for k in ('token_hit', 'norm_hit', 'any_hit', 'right'):
                c[k + '_rate'] = round(c[k] / c['n'], 4)
            c['sec'] = round(c['sec'] / c['n'], 3)
            res[f'{task}/{keys}'] = dict(c, misses=miss[:8])
            print(f"{task:5s} {keys:9s} token {c['token_hit_rate']:.3f} norm {c['norm_hit_rate']:.3f} "
                  f"(s1 {c['step1']} s2 {c['step2']} s3 {c['step3']}) any {c['any_hit_rate']:.3f} "
                  f"right {c['right_rate']:.3f}  {c['sec']}s/doc", flush=True)
            for m in miss[:3]:
                print('   miss', json.dumps(m)[:300], flush=True)
    json.dump(dict(args=vars(a), cells=res), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
