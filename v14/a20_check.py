"""A20 (pre-registered 2026-11-01 10:32 IST): the rare-in-English unigram step on reworded questions, text only (CPU).

bench's own docs (build(..., qstyle='varied')), seed 37000, 32k, 40 docs x kv / kv1 / kvupd / neg x synthetic /
natural: the current chain (text_fetch) vs the chain with step 4 (text_fetch(step4=True)), on the same docs.

    python v14/a20_check.py --out logs/v14_a20_check_<hash>_<stamp>.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from memory_probes.spread import filler_tokens  # noqa: E402
from v16.bench import build  # noqa: E402
from v7.data import get_chat_tokenizer  # noqa: E402
from v14 import norm_organ as N  # noqa: E402
from v14 import package_fetch as P  # noqa: E402
from v14.styles import QUESTIONS  # noqa: E402

FILL = '/home/gowrav/Development/qllm2/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'


def style_of(q):
    for name, t in QUESTIONS:
        pre, post = t.split('{k}')
        if q.startswith(pre) and q.endswith(post):
            return name
    return '?'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=37000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=40); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tok = get_chat_tokenizer(); P._S['strs'] = N.tok_strings(tok)
    fill = filler_tokens(FILL)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    res = {}
    for keys in ('synthetic', 'natural'):
        for task in ('kv', 'kv1', 'kvupd', 'neg'):
            rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(task.encode()) % 997)   # = bench.py main()
            c = collections.Counter(); per = collections.defaultdict(collections.Counter)
            for d in range(a.docs):
                ids, ans, qlen, meta = build(rng, tok, filler, a.L, task, 8, keys=keys, qstyle='varied')
                end = len(ids) - qlen; q = tok.decode(ids[end:]); s = style_of(q)
                fact = f"{meta['key']} refers to {meta['val']}"
                for arm, kw in (('old', {}), ('new', {'step4': True})):
                    row, info = P.text_fetch(ids, end, **kw)
                    right = row is not None and not meta.get('negative') and fact in tok.decode(ids[row[0]:row[1]])
                    c[arm + '_hit'] += info['organ_hit']; c[arm + '_right'] += int(right)
                    if arm == 'new':
                        c[f"new_step{info['step']}"] += 1
                        per[s]['n'] += 1; per[s]['right'] += int(right); per[s]['hit'] += info['organ_hit']
                c['n'] += 1
            n = c['n']
            res[f'{task}/{keys}'] = dict({k: round(v / n, 4) if k != 'n' else v for k, v in c.items()},
                                         per_wording={k: dict(v) for k, v in per.items()})
            print(f"{task:5s} {keys:9s} old hit {c['old_hit']/n:.3f} right {c['old_right']/n:.3f} | new hit {c['new_hit']/n:.3f} "
                  f"right {c['new_right']/n:.3f} | new steps {dict((k[8:], v) for k, v in c.items() if k.startswith('new_step'))}", flush=True)
    r = res
    bars = dict(syn_kv=r['kv/synthetic']['new_right'] >= 0.90,
                neg=max(r['neg/synthetic']['new_hit'], r['neg/natural']['new_hit']) <= 0.05,
                nat_kv=r['kv/natural']['new_right'] >= r['kv/natural']['old_right'] - 0.05)
    res['bars'] = bars; res['verdict'] = 'MET' if all(bars.values()) else 'NOT met'
    print('BARS', bars, '->', res['verdict'], flush=True)
    json.dump(dict(args=vars(a), cells=res), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
