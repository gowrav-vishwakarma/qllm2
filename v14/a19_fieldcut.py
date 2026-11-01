"""A19 (pre-registered 2026-11-01 10:10 IST): field-cut tool rows, a labelled PLUMBING row ("field-cut rows").

Lead's text-arm tool docs (seed 18000, 32k, 40 docs x synthetic / natural); row = text_fetch (strict); cut =
package_fetch.field_cut (the asked field's clause, re-rendered). --part cold (CPU): b2p2 and T++ cold on cut and uncut
rows. --part warm (GPU): b2p2 warm + no-row contrast on cut and uncut rows.

    python v14/a19_fieldcut.py --part cold --device cpu --out logs/v14_a19_fieldcut_cold_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
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
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from v16.bench import Reader, build  # noqa: E402
from v14.a6_reread import CKPT, FILL, Mind, sent_ok  # noqa: E402
from v14.a18_tpp_warm import TPP, tpp_greedy  # noqa: E402
from v14.package_fetch import field_cut, text_fetch  # noqa: E402

ALPHA = 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['cold', 'warm'], required=True)
    ap.add_argument('--seed', type=int, default=18000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=40); ap.add_argument('--needles', type=int, default=8)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    pam = Mind(CKPT, dev); tok = pam.tok
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    R = None
    if a.part == 'cold':
        tm, _, _ = _load_v13_sempty(Path(TPP), 'baseline_real_pm', dev); tm.eval()
        R = Reader(tm, 'transformer')
    fill = filler_tokens(FILL)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    outf = open(a.out, 'a')
    for keys in ('synthetic', 'natural'):
        rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(b'tool') % 997)       # = bench.py main() for task tool
        for d in range(a.docs):
            t0 = time.time()
            ids, ans, qlen, meta = build(rng, tok, filler, a.L, 'tool', a.needles, keys=keys)
            end = len(ids) - qlen; q = ids[end:]; val = str(meta['val'])
            row, info = text_fetch(ids, end)
            rec = dict(part=a.part, task='tool', keys=keys, seed=a.seed, doc=d, L=a.L, abstain=info['abstain'])
            if row is not None:
                span = ids[row[0]:row[1]]
                cut_text = field_cut(tok.decode(span), tok.decode(q))
                rec['cut_ok'] = int(cut_text is not None); rec['cut_text'] = cut_text
                cut = enc(cut_text) if cut_text else span
                rows = {'uncut': span, 'cut': cut}
                if a.part == 'cold':
                    for name, r in rows.items():
                        for body, g in (('b2p2', lambda x: pam.greedy(x)), ('tpp', lambda x: tpp_greedy(R, x))):
                            txt = tok.decode(g(r + q))
                            rec[f'{body}_cold_{name}'] = sent_ok(txt, val); rec[f'{body}_cold_{name}_text'] = txt
                else:
                    _, st, off = pam.feed(ids[:end])
                    for name, r in rows.items():
                        g = pam.greedy_contrast(r + q, copy.deepcopy(st), off, q, copy.deepcopy(st), off, ALPHA)
                        txt = tok.decode(g)
                        rec[f'b2p2_warmc_{name}'] = sent_ok(txt, val); rec[f'b2p2_warmc_{name}_text'] = txt
            rec['sec'] = round(time.time() - t0, 1)
            outf.write(json.dumps(rec) + '\n'); outf.flush()
            print(keys, d, rec.get('cut_ok'), {k: rec[k] for k in rec if k.startswith(('b2p2_', 'tpp_')) and not k.endswith('text')},
                  rec['sec'], flush=True)


if __name__ == '__main__':
    main()
