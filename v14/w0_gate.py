"""W0 (eval only): does the body's own surprise find the events worth a library row?

    .venv/bin/python -m v14.w0_gate --checkpoint <best_model.pt> --out logs/v14_w0_<hash>_<time>.json

Pre-registered in v14/EXPERIMENTS_V14.md, section "L" (unit W0 + amendment A1).
A frozen v13 PAM LM reads documents of L tokens of held-out prose with 8 needles inserted.
Surprise u_t = -log p(x_t | x_<t) comes from the body's PRE-MEMORY vector
(head projection + norm, before the echo read is added), through the tied head; the
echo-included surprise is reported for reference. A token with u_t > tau is a row.
Cells: synthetic definitions, natural keys (the organiser's definition template with
real two-word phrases), and plain real sentences (reported against the chance rate).
Metrics per tau: row rate on filler, share of needles with a firing token inside the
needle sentence, and where in the sentence the firing token sits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from memory_probes.spread import filler_tokens  # noqa: E402
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
try:
    from v16.bench import natural_key  # noqa: E402  (the harness moved here at d1e1a56)
except ImportError:                      # older checkouts
    from session_index_probe import natural_key  # noqa: E402
from v7.data import _RECALL_VALUE_NOUNS, _recall_key  # noqa: E402

TAUS = (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
HOLDOUT = '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'
DOT = 13


@torch.no_grad()
def surprises(model, ids: list[int], device) -> tuple[np.ndarray, np.ndarray]:
    """(u_pre, u_full) [T-1]: surprise of token t+1 from the vector at t, without / with echo."""
    x = torch.tensor([ids], device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, b, t = model._stem(x, 0)
        z, _ = model._run_blocks(z, b, t, None, 0, x)
        hidden = model.output_norm(z)
        pre = model.lm_head_norm(model.lm_head_proj(hidden))
        echo, _ = model._echo_read(hidden, x, b, t, None)
        full = pre + echo
        pre = pre.raw(b, t, model.model_dim)[0]
        full = full.raw(b, t, model.model_dim)[0]
    E = model.embed.embed.weight.float()
    tgt = x[0, 1:]
    out = []
    for v in (pre, full):
        u = torch.empty(len(ids) - 1, device=device)
        for s in range(0, len(ids) - 1, 1024):
            lg = v[s:min(s + 1024, len(ids) - 1)].float() @ E.t()
            u[s:s + lg.shape[0]] = torch.logsumexp(lg, -1) - lg.gather(-1, tgt[s:s + lg.shape[0], None]).squeeze(-1)
        out.append(u.cpu().numpy())
    return out[0], out[1]


def make_doc(rng, tok, filler, L, cell, n_needles=8):
    """-> ids, needles [(start, end, key_tokens range, value tokens range)]; positions index the final ids."""
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    start = rng.randrange(0, len(filler) - L - 1)
    body = filler[start:start + L].tolist()
    pieces = []  # (pos in body, ids, key_len, val_len)
    for _ in range(n_needles):
        if cell == 'synthetic':
            k, v = _recall_key(rng), rng.choice(_RECALL_VALUE_NOUNS)
            ids = enc(f" Definition: {k} refers to {v}.")
            kl, vl = len(enc(f" {k}")), len(enc(f" {v}"))
        elif cell == 'natural':
            k, v = natural_key(rng, tok, filler, start, start + L), rng.choice(_RECALL_VALUE_NOUNS)
            ids = enc(f" Definition: {k} refers to {v}.")
            kl, vl = len(enc(f" {k}")), len(enc(f" {v}"))
        else:  # plain: a real sentence from outside the document's slice
            while True:
                i = rng.randrange(0, len(filler) - 64)
                if start - 64 <= i <= start + L:
                    continue
                seg = filler[i:i + 64].tolist()
                dots = [j for j, t in enumerate(seg) if t == DOT]
                if len(dots) >= 2 and 8 <= dots[1] - dots[0] <= 40:
                    ids = seg[dots[0] + 1:dots[1] + 1]
                    break
            kl = vl = 0
        pieces.append((rng.randrange(0, L - 64), ids, kl, vl))
    pieces.sort(key=lambda p: p[0])
    ids, needles, cur = [], [], 0
    for pos, nid, kl, vl in pieces:
        ids += body[cur:pos]
        s = len(ids)
        ids += nid
        needles.append((s, len(ids), kl, vl))
        cur = pos
    ids += body[cur:]
    return ids, needles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--length', type=int, default=8192)
    ap.add_argument('--docs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=9100)
    ap.add_argument('--mem_frac', type=float, default=0.0)
    args = ap.parse_args()
    if args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    device = torch.device('cuda')
    model, _, _ = _load_v13_sempty(args.checkpoint, 'baseline_real_pm', device)
    model.eval()
    filler = filler_tokens(str(ROOT / HOLDOUT))
    rng = random.Random(args.seed)
    res = {'checkpoint': args.checkpoint, 'length': args.length, 'docs': args.docs, 'taus': TAUS,
           'seed': args.seed, 'cells': {}}
    for cell in ('synthetic', 'natural', 'plain'):
        acc = {m: {t: dict(rows=0, toks=0, caught=0, needles=0, key=0, val=0, first=0, len_sum=0)
                   for t in TAUS} for m in ('pre', 'full')}
        for d in range(args.docs):
            ids, needles = make_doc(rng, tok, filler, args.length, cell)
            up, uf = surprises(model, ids, device)        # u[j] = surprise of token j+1
            for name, u in (('pre', up), ('full', uf)):
                inside = np.zeros(len(ids), dtype=bool)
                for s, e, _, _ in needles:
                    inside[s:e] = True
                for t in TAUS:
                    fire = np.zeros(len(ids), dtype=bool)
                    fire[1:] = u > t
                    a = acc[name][t]
                    a['rows'] += int(fire[~inside].sum())
                    a['toks'] += int((~inside).sum())
                    for s, e, kl, vl in needles:
                        a['needles'] += 1
                        a['len_sum'] += e - s
                        hit = fire[s:e]
                        a['caught'] += int(hit.any())
                        if cell != 'plain':
                            # tokens: " Definition" ":" key... " refers" " to" value... "."
                            ks = s + 2                      # after " Definition", ":"
                            a['key'] += int(fire[ks:ks + kl].any())
                            vs = e - 1 - vl
                            a['val'] += int(fire[vs:e - 1].any())
                        a['first'] += int(hit[0])
            print(f"  [{cell}] doc {d + 1}/{args.docs}", flush=True, end='\r')
        out = {}
        for name in ('pre', 'full'):
            out[name] = {}
            for t in TAUS:
                a = acc[name][t]
                r = a['rows'] / max(1, a['toks'])
                mean_len = a['len_sum'] / max(1, a['needles'])
                out[name][str(t)] = dict(
                    row_rate=r, rows_per_16=16 * r, caught=a['caught'] / max(1, a['needles']),
                    key_fires=a['key'] / max(1, a['needles']), value_fires=a['val'] / max(1, a['needles']),
                    chance_catch=1 - (1 - r) ** mean_len, needles=a['needles'])
        res['cells'][cell] = out
        print(f"\n[{cell}] pre-memory surprise (tau: rows/16 tok | needle caught | key | value | chance):")
        for t in TAUS:
            o = out['pre'][str(t)]
            print(f"  tau {t:4.1f}: {o['rows_per_16']:.2f} | {o['caught']:.3f} | {o['key_fires']:.3f} | "
                  f"{o['value_fires']:.3f} | {o['chance_catch']:.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1)
    print('saved', args.out)


if __name__ == '__main__':
    main()
