"""A16/1 (amendment 2026-11-01 08:44 IST, Lead's ask): the PROSE cost of echo dilation over a refresh gap.

For A16's docs (Mind's skill pack, seeds 8600-8649, L 32k, R 8192): the mean NLL of the filler tokens in the last
gap (after the last pack read, before the question) with vs without echo dilation s = 4 on that gap.

    python -m v14.a16_span_nll --out logs/v14_a16_span_nll_<hash>_<stamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))

from memory_probes.spread import _SEP, filler_tokens  # noqa: E402
from run_skillpack_probe import filler_block  # noqa: E402
from v14.n11_refresh import B2P2, last_gap, stream  # noqa: E402
from v14.sr0 import HOLDOUT, Mind, doc  # noqa: E402


@torch.no_grad()
def span_nll(mind, ids, a, b, dil):
    """mean NLL (nats) of ids[a:b] given their prefixes, with model._dilate = dil."""
    m = mind.m
    m._dilate = dil
    try:
        x = torch.tensor([ids], device=mind.dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z, bd, td = m._stem(x, 0)
            shared = None
            for blk in m.blocks:
                if m.config.share_reads:
                    z, _ = blk(z, pam_state=None, step_offset=0, shared=shared)
                    if blk.pam is not None:
                        shared = blk.pam._shared
                else:
                    z, _ = blk(z, pam_state=None, step_offset=0)
            lm, _ = m._lm_vector(z, x, bd, td)
        h = lm.raw(bd, td, m.model_dim)[0, a - 1:b - 1].float()
    finally:
        m._dilate = None
    tgt = x[0, a:b]
    tot = 0.0
    for s in range(0, h.shape[0], 2048):
        lg = h[s:s + 2048] @ mind.E.t()
        tot += float(torch.nn.functional.cross_entropy(lg.float(), tgt[s:s + 2048], reduction='sum'))
    return tot / (b - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='8600,8650'); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--R', type=int, default=8192); ap.add_argument('--dilate', type=float, default=4.0)
    ap.add_argument('--tasks', default='affine_map_new,shift_wide_new'); ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    mind = Mind(B2P2, torch.device(a.device)); tok = mind.tok
    filler = filler_tokens(HOLDOUT); sep = tok.encode(_SEP, add_special_tokens=False)
    s0, s1 = (int(x) for x in a.seeds.split(','))
    rows = []
    for task in a.tasks.split(','):
        for seed in range(s0, s1):
            ex, H, U, pack, q = doc(tok, task, seed)
            fill = filler_block(filler, a.L, seed, a.L)
            ids, _ = stream(pack, fill, q, sep, a.R)
            g0, _ = last_gap(pack, fill, sep, a.R)
            e = len(ids) - len(q)
            plain = span_nll(mind, ids, g0, e, None)
            dil = span_nll(mind, ids, g0, e, (g0, e, a.dilate, 'echo'))
            rows.append(dict(task=task, seed=seed, span=e - g0, nll_plain=plain, nll_dilated=dil, cost=dil - plain))
            print(task, seed, e - g0, round(plain, 4), round(dil, 4), round(dil - plain, 4), flush=True)
    c = [r['cost'] for r in rows]
    summary = dict(n=len(rows), mean_cost_nats=sum(c) / len(c), mean_plain=sum(r['nll_plain'] for r in rows) / len(rows))
    print(json.dumps(summary), flush=True)
    json.dump(dict(args=vars(a), summary=summary, rows=rows), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
