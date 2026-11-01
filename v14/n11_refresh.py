"""N11-R: the planner's skill-refresh rule (v14/EXPERIMENTS_V14.md "N11-R", pre-registered 23:32 IST). Eval only.

Mind's skill-pack cell on b2p2: the pack (header + 8 demonstration lines) is read at t0, then L tokens of held-out
filler in which the pack is RE-INSERTED after every R filler tokens, then the query line. R = 0 means no refresh.
Scored at the last position: the 20-way pick (correct, margin) as scripts/run_skillpack_probe.py does, and FREE
generation (the greedy token over the whole vocabulary equals the target; primary since the user's 'generation counts').

    .venv/bin/python -m v14.n11_refresh --out <jsonl>        (seeds 8100-8149, R 1024/2048/4096/0, L 8192/32768)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from memory_probes.spread import _SEP, filler_tokens  # noqa: E402
from run_skillpack_probe import filler_block  # noqa: E402
from v14.sr0 import HOLDOUT, Mind, doc, score  # noqa: E402

B2P2 = '/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt'


def stream(pack, fill, q, sep, R):
    out = list(pack)
    if R <= 0:
        return out + sep + fill + sep + q, 0
    n_ins = 0
    for i in range(0, len(fill), R):
        out += sep + fill[i:i + R]
        if i + R < len(fill):
            out += sep + pack
            n_ins += 1
    return out + sep + q, n_ins


def last_gap(pack, fill, sep, R):
    """(start of the last filler chunk in stream()'s ids, its offset in fill): the stretch after the last pack read."""
    if R <= 0:
        return len(pack) + len(sep), 0
    n, start, off = len(pack), None, 0
    for i in range(0, len(fill), R):
        n += len(sep); start, off = n, i; n += len(fill[i:i + R])
        if i + R < len(fill):
            n += len(sep) + len(pack)
    return start, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=B2P2)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', default='affine_map_new,shift_wide_new')
    ap.add_argument('--seeds', default='8100,8150')
    ap.add_argument('--refresh', default='1024,2048,4096,0')
    ap.add_argument('--lengths', default='8192,32768')
    ap.add_argument('--mem_frac', type=float, default=0.35)
    ap.add_argument('--contrast', type=float, default=None,
                    help='A8 (N11-RC): also score l_R + a (l_R - l_nopack), nopack = filler + question, no pack')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dilate', type=float, default=None,
                    help='A16: echo dilation s (retention ** 1/s) from the last pack read to the question, with the contrast')
    ap.add_argument('--dilate_refresh', default='', help='A16: the R values that also get the dilated + contrast arm')
    a = ap.parse_args()
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    mind = Mind(a.ckpt, torch.device(a.device))
    tok = mind.tok
    filler = filler_tokens(HOLDOUT)
    sep = tok.encode(_SEP, add_special_tokens=False)
    s0, s1 = (int(x) for x in a.seeds.split(','))
    out = open(a.out, 'a')
    for task in a.tasks.split(','):
        for seed in range(s0, s1):
            ex, H, U, pack, q = doc(tok, task, seed)
            for L in (int(x) for x in a.lengths.split(',')):
                fill = filler_block(filler, L, seed, L)
                lg_n = mind.forward(sep + fill + sep + q)[0] if a.contrast is not None else None
                for R in (int(x) for x in a.refresh.split(',')):
                    t0 = time.time()
                    ids, n_ins = stream(pack, fill, q, sep, R)
                    lg = mind.forward(ids)[0]
                    c, m = score(ex, lg)
                    top = int(lg.argmax())                  # free generation: greedy token over the whole vocab
                    extra = {}
                    if lg_n is not None:                    # A8: state-contrast decoding at the query
                        lgc = lg + a.contrast * (lg - lg_n)
                        cc, mc = score(ex, lgc); tc = int(lgc.argmax())
                        extra = dict(alpha=a.contrast, correct_c=cc, margin_c=mc,
                                     free_c=int(tc == ex.target_token_id), free_tok_c=tok.decode([tc]))
                    if a.dilate is not None and str(R) in a.dilate_refresh.split(','):
                        g0, off = last_gap(pack, fill, sep, R)                     # A16: echo dilation, then contrast
                        nq = len(sep) + len(fill) + len(sep)
                        mind.m._dilate = (g0, len(ids) - len(q), a.dilate, 'echo')
                        lg_d = mind.forward(ids)[0]
                        mind.m._dilate = (len(sep) + off, nq, a.dilate, 'echo')
                        lg_nd = mind.forward(sep + fill + sep + q)[0]
                        mind.m._dilate = None
                        lgdc = lg_d + (a.contrast or 0.0) * (lg_d - lg_nd)
                        cd, md = score(ex, lgdc); td = int(lgdc.argmax())
                        extra.update(dilate=a.dilate, dilate_span=len(ids) - len(q) - g0, correct_dc=cd, margin_dc=md,
                                     free_dc=int(td == ex.target_token_id), free_tok_dc=tok.decode([td]),
                                     free_d=int(int(lg_d.argmax()) == ex.target_token_id))
                    out.write(json.dumps(dict(task=task, doc=seed, L=L, R=R, correct=c, margin=m,
                                              free=int(top == ex.target_token_id), free_tok=tok.decode([top]),
                                              inserts=n_ins,
                                              pack_tokens=len(pack), total_tokens=len(ids),
                                              overhead=round(n_ins * len(pack) / max(1, L), 4),
                                              sec=round(time.time() - t0, 2), **extra)) + '\n')
                    out.flush()
        print(f'{task} done', flush=True)


if __name__ == '__main__':
    main()
