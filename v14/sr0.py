"""SR-0 (v14/EXPERIMENTS_V14.md "SR", amendments A1/A2): is there a transplantable rule vector?

Mind's skill-pack cell (scripts/run_skillpack_probe.py): B2 rules shift_wide_new / affine_map_new, 8 demos.
  A  : pack read once at t0, L tokens of filler, then the query (the body's own reach; the floor).
  B  : NO pack: header + filler + query, with the rule vector f added to the residual stream after block l at
       the query's last position (the position that predicts the answer).
  C  : as B with the vector of a DIFFERENT rule (another document's pack whose rule parameter differs).
  D  : as B with a random vector of the same norm.
f = the mean residual after block l at the arrow tokens of the pack's last 4 demonstrations (the mind is about
to write each demo's answer there; averaging 4 demos keeps the rule and dilutes any one answer).
Each arm is a full forward pass (b2p2 is an M2 body with shared reads, which has no decode path).

    .venv/bin/python -m v14.sr0 dev  --ckpt <b2p2> --out <json>      # choose layer and scale on seeds 8200-8219
    .venv/bin/python -m v14.sr0 test --ckpt <b2p2> --layer L --scale S --out <jsonl>   # seeds 8000-8099
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from memory_probes.rules import score_candidate_logits  # noqa: E402
from memory_probes.spread import _SEP, filler_tokens, split_units  # noqa: E402
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from run_skillpack_probe import filler_block  # noqa: E402
from v14.rules_ood import build_example as b2_example  # noqa: E402

HOLDOUT = str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')


class Mind:
    def __init__(self, ckpt, dev):
        self.m, self.tok, _ = _load_v13_sempty(ckpt, 'baseline_real_pm', dev)
        self.m.eval()
        self.dev = dev
        self.E = self.m.embed.embed.weight.float()

    @torch.no_grad()
    def forward(self, ids, layer=None, add=None, capture=None):
        """Full forward (M2 shared reads need it: no decode). Returns (last-position logits, residual after
        block `capture` at every position or None). With `add`, it is added to the residual after block
        `layer` at the LAST position only."""
        from sempyt.tensor import named
        m = self.m
        x = torch.tensor([ids], device=self.dev)
        cap = None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z, b, t = m._stem(x, 0)
            shared = None
            for i, blk in enumerate(m.blocks):
                if m.config.share_reads:
                    z, _ = blk(z, pam_state=None, step_offset=0, shared=shared)
                    if blk.pam is not None:
                        shared = blk.pam._shared
                else:
                    z, _ = blk(z, pam_state=None, step_offset=0)
                if capture is not None and i + 1 == capture:
                    cap = z.raw(b, t, m.model_dim)[0].float()
                if add is not None and i + 1 == layer:
                    raw = z.raw(b, t, m.model_dim).clone()
                    raw[0, -1] = raw[0, -1] + add.to(raw.dtype)
                    z = named(raw, (b, t, m.model_dim))
            lm, _ = m._lm_vector(z, x, b, t)
            v = lm.raw(b, t, m.model_dim)[0, -1].float()
        return v @ self.E.t(), cap


def doc(tok, task, seed):
    ex = b2_example(tok, task, 8, seed)
    header, units = split_units(ex)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    H, U = enc(header), [enc(u) for u in units]
    pack = H + sum(U[:-1], [])
    return ex, H, U, pack, U[-1]


def rule_vector(mind, tok, pack, layer):
    """mean residual after `layer` at the arrow tokens of the last 4 demonstrations of the pack."""
    arrow = tok.encode(' ->', add_special_tokens=False)[-1]
    pos = [i for i, t in enumerate(pack) if t == arrow][-4:]
    _, r = mind.forward(pack, capture=layer)
    return r[pos].mean(0)


def score(ex, logits):
    r = score_candidate_logits(ex, logits[ex.candidate_token_ids].float().cpu().tolist())
    return int(r['correct']), float(r['target_margin'])


def param(ex):
    return str(getattr(ex, 'meta', None) or ex.target_text)


def run_cell(mind, tok, filler, sep, task, seed, L, layer, scales, other):
    """-> dict of arm -> (correct, margin) for this doc and gap L."""
    ex, H, U, pack, q = doc(tok, task, seed)
    fill = sep + filler_block(filler, L, seed, L) + sep if L else []
    out = {'A': score(ex, mind.forward(pack + fill + q)[0])}      # pack at t0, filler, query
    ids_b = H + fill + q                                            # no pack
    out['none'] = score(ex, mind.forward(ids_b)[0])
    f = rule_vector(mind, tok, pack, layer)
    g = rule_vector(mind, tok, other, layer)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    rnd = torch.randn(f.shape, generator=gen).to(f.device)
    rnd = rnd / rnd.norm() * f.norm()
    for s in scales:
        out[f'B@{s}'] = score(ex, mind.forward(ids_b, layer, s * f)[0])
        out[f'C@{s}'] = score(ex, mind.forward(ids_b, layer, s * g)[0])
        out[f'D@{s}'] = score(ex, mind.forward(ids_b, layer, s * rnd)[0])
    return out


def other_pack(tok, task, seed):
    """a pack of the same task whose answer for this doc's query differs (a different rule parameter)."""
    ex, *_ = doc(tok, task, seed)
    for k in range(1, 50):
        ex2, H2, U2, pack2, q2 = doc(tok, task, seed + 100000 + k)
        if ex2.target_text != ex.target_text or ex2.prompt_text.split('\n')[1] != ex.prompt_text.split('\n')[1]:
            return pack2
    return pack2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['dev', 'test'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--layer', type=int)
    ap.add_argument('--scale', type=float)
    ap.add_argument('--tasks', default='affine_map_new,shift_wide_new')
    ap.add_argument('--lengths', default='0,2048,4096,8192,32768,131072')
    ap.add_argument('--docs', type=int, default=100)
    ap.add_argument('--mem_frac', type=float, default=0.3)
    a = ap.parse_args()
    dev = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    mind = Mind(a.ckpt, dev)
    tok = mind.tok
    filler = filler_tokens(HOLDOUT)
    sep = tok.encode(_SEP, add_special_tokens=False)
    tasks = a.tasks.split(',')
    if a.mode == 'dev':
        # choose (layer, scale) on seeds 8200-8219 at L = 8192 (where the body has faded): best mean B accuracy
        grid_l, grid_s = (2, 4, 6, 8, 10, 12, 14), (1.0, 2.0, 4.0, 8.0)
        res = {}
        for l in grid_l:
            acc = {s: [] for s in grid_s}
            for task in tasks:
                for seed in range(8200, 8220):
                    o = run_cell(mind, tok, filler, sep, task, seed, 8192, l, grid_s, other_pack(tok, task, seed))
                    for s in grid_s:
                        acc[s].append(o[f'B@{s}'][0])
            res[l] = {s: float(np.mean(v)) for s, v in acc.items()}
            print(f"layer {l}: " + ' '.join(f"x{s}: {res[l][s]:.2f}" for s in grid_s), flush=True)
        best = max(((l, s) for l in grid_l for s in grid_s), key=lambda ls: res[ls[0]][ls[1]])
        json.dump({'grid': {str(l): v for l, v in res.items()}, 'best_layer': best[0], 'best_scale': best[1]},
                  open(a.out, 'w'), indent=1)
        print('best', best, flush=True)
        return
    out = open(a.out, 'a')
    lengths = [int(x) for x in a.lengths.split(',')]
    for task in tasks:
        for seed in range(8000, 8000 + a.docs):
            other = other_pack(tok, task, seed)
            for L in lengths:
                t0 = time.time()
                o = run_cell(mind, tok, filler, sep, task, seed, L, a.layer, (a.scale,), other)
                out.write(json.dumps({'task': task, 'doc': seed, 'L': L, 'layer': a.layer, 'scale': a.scale,
                                      **{k: {'correct': v[0], 'margin': v[1]} for k, v in o.items()},
                                      'sec': round(time.time() - t0, 2)}) + '\n')
                out.flush()
        print(f'{task} done', flush=True)


if __name__ == '__main__':
    main()
