"""A6 (pre-registered 2026-11-01 01:30 IST): hand the fetched row WARM (the mind's carried state after the
document) or COLD (fresh state, row + question alone, as bench.py's re-read). Fetch is held fixed: the row is
the sentence holding the asked fact, found by construction. Score: Lead's sentence scorer (greedy to '.' /
newline, cap 16, the value as a whole word, case-insensitive).

    python v14/a6_reread.py --parity --device cpu                       # state path == one-pass logits
    python v14/a6_reread.py --device cuda --seed 26000 --L 32768 --docs 40 --out logs/v14_a6_<hash>_<stamp>.jsonl
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
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from memory_probes.spread import filler_tokens  # noqa: E402
from v16.bench import build  # noqa: E402
from v14 import package_fetch as P  # noqa: E402

CKPT = '/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt'
FILL = '/home/gowrav/Development/qllm2/.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'
STOPS = (13, 198)
CHUNK = 2048


class Mind:
    def __init__(self, ckpt, dev):
        self.m, self.tok, _ = _load_v13_sempty(Path(ckpt), 'baseline_real_pm', dev)
        self.m.eval(); self.dev = dev

    @torch.no_grad()
    def feed(self, ids, states=None, off=0, chunk=CHUNK):
        """run ids from (states, off); returns (last logits, new states, new offset)."""
        lg = None
        for s in range(0, len(ids), chunk):
            x = torch.tensor([ids[s:s + chunk]], device=self.dev)
            logits, states, _ = self.m(x, states=states, step_offset=off + s)
            lg = logits[0, -1].float()
        return lg, states, off + len(ids)

    @torch.no_grad()
    def greedy(self, prefix, states=None, off=0, max_new=16):
        lg, st, off = self.feed(prefix, states, off)
        out = []
        for _ in range(max_new):
            t = int(lg.argmax()); out.append(t)
            if t in STOPS:
                break
            lg, st, off = self.feed([t], st, off)
        return out


def fact_row(ids, end, tok, meta):
    """(start, end) of the sentence holding the asked fact: the key's tokens within 24 tokens of the value's."""
    kt = tok.encode(f" {meta['key']}", add_special_tokens=False)
    vt = tok.encode(f" {meta['val']}", add_special_tokens=False)
    def occ(t):
        return [i for i in range(end - len(t) + 1) if ids[i] == t[0] and ids[i:i + len(t)] == t]
    vs = occ(vt)
    best = None
    for k in occ(kt):
        for v in vs:
            if abs(v - k) <= 24:
                best = (min(k, v), max(k + len(kt), v + len(vt)))
    return P.row_around(ids, best[0], best[1], end) if best else None


def sent_ok(text, val):
    return int(re.search(r'\b' + re.escape(str(val)) + r'\b', text, re.I) is not None)


def parity(mind, filler):
    """the chunked carried-state path must equal one pass (bench's Reader runs one pass)."""
    rng = random.Random(7)
    ids, ans, qlen, meta = build(rng, mind.tok, filler, 4096, 'kv', 8, keys='synthetic')
    with torch.no_grad():
        full = mind.m(torch.tensor([ids], device=mind.dev))[0][0, -1].float()
    for ch in (1024, 777):
        lg, _, _ = mind.feed(ids, chunk=ch)
        print(f'parity chunk {ch}: max |diff| {float((lg - full).abs().max()):.2e}  argmax equal {int(lg.argmax()) == int(full.argmax())}')
    end = len(ids) - qlen
    lg_a, st, off = mind.feed(ids[:end])
    lg_b, _, _ = mind.feed(ids[end:], st, off)
    print(f'parity doc-state + question: max |diff| {float((lg_b - full).abs().max()):.2e}')
    row = fact_row(ids, end, mind.tok, meta)
    span = ids[row[0]:row[1]] + ids[end:]
    g1 = mind.greedy(span)
    g2 = []
    for _ in range(16):                                     # bench.py Reader.greedy: re-run the context each step
        with torch.no_grad():
            t = int(mind.m(torch.tensor([span + g2], device=mind.dev))[0][0, -1].argmax())
        g2.append(t)
        if t in STOPS:
            break
    print('parity cold greedy (state path vs re-run):', g1 == g2, repr(mind.tok.decode(g1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT); ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=26000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=40); ap.add_argument('--needles', type=int, default=8)
    ap.add_argument('--tasks', default='kv,para,tool,natq'); ap.add_argument('--keys', default='synthetic,natural')
    ap.add_argument('--parity', action='store_true'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    mind = Mind(a.ckpt, dev)
    fill = filler_tokens(FILL)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    if a.parity:
        parity(mind, filler)
        return
    outf = open(a.out, 'a')
    for keys in a.keys.split(','):
        for task in a.tasks.split(','):
            rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(task.encode()) % 997)   # = bench.py main()
            for d in range(a.docs):
                t0 = time.time()
                ids, ans, qlen, meta = build(rng, mind.tok, filler, a.L, task, a.needles, keys=keys)
                end = len(ids) - qlen; q = ids[end:]
                row = fact_row(ids, end, mind.tok, meta)
                rec = dict(task=task, keys=keys, L=a.L, seed=a.seed, doc=d, val=meta['val'], row_found=int(row is not None))
                if row is not None:
                    span = ids[row[0]:row[1]]
                    _, st, off = mind.feed(ids[:end])
                    cold = mind.greedy(span + q)
                    warm = mind.greedy(span + q, copy.deepcopy(st), off)
                    norow = mind.greedy(q, st, off)
                    for name, g in (('cold', cold), ('warm', warm), ('norow', norow)):
                        txt = mind.tok.decode(g)
                        rec[name] = sent_ok(txt, meta['val']); rec[name + '_text'] = txt
                    rec['row_text'] = mind.tok.decode(span)[-160:]
                rec['sec'] = round(time.time() - t0, 2)
                outf.write(json.dumps(rec) + '\n'); outf.flush()
                print(task, keys, d, {k: rec.get(k) for k in ('row_found', 'cold', 'warm', 'norow')}, rec['sec'], flush=True)


if __name__ == '__main__':
    main()
