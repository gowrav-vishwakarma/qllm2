#!/usr/bin/env python3
"""Minimal standalone repro of the checkpoint save-order swap in cnormalize_vec.

Isolates the div `x / mag.unsqueeze(-1).unsqueeze(-1)` and runs it through the
exact non-reentrant checkpoint pattern (fwd under no_grad, recompute under
grad). Compares the SAVED operand order (fwd) vs the RECOMPUTED operand order
(recompute) to confirm the mechanism, then tests candidate fixes:

  F0  x / mag_uu                     (current: view denominator)
  F1  x / mag_uu.contiguous()        (materialized denominator)
  F2  x * (1.0 / mag_uu)             (reciprocal, then mul)
  F3  x * mag_uu.reciprocal()        (reciprocal as a tensor)
  F4  torch.div(x, mag_uu)           (explicit div)

A fix PASSES if the saved-operand order is stable between the two passes.

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_minrep
"""
from __future__ import annotations
import torch
import torch.utils.checkpoint as ckpt

B, H, T, D = 2, 2, 128, 32
torch.manual_seed(0)


def make_fn(kind):
    def cnorm(x):
        mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
        m = mag.unsqueeze(-1).unsqueeze(-1)
        if kind == 'F0':
            return x / m
        if kind == 'F1':
            return x / m.contiguous()
        if kind == 'F2':
            return x * (1.0 / m)
        if kind == 'F3':
            return x * m.reciprocal()
        if kind == 'F4':
            return torch.div(x, m)
        raise ValueError(kind)
    return cnorm


# capture saved-tensor shape order in fwd and in recompute
_orig_meta = ckpt._allowed_determinism_checks_to_fns['default']
FWD_SHAPES, RECOMP_SHAPES = [], []
REGION = {'cur': None}


def _meta_rec(x):
    meta = _orig_meta(x)
    if REGION['cur'] == 'fwd':
        FWD_SHAPES.append(tuple(x.shape))
    return meta


def _recompute_rec(x):
    # called by _recomputation_hook? no — use saved_tensors_hooks around recompute.
    RECOMP_SHAPES.append(tuple(x.shape))
    return x


ckpt._allowed_determinism_checks_to_fns['default'] = _meta_rec


def run_case(kind):
    FWD_SHAPES.clear()
    RECOMP_SHAPES.clear()
    fn = make_fn(kind)
    x = torch.randn(B, H, T, D, 2, device='cuda', requires_grad=True)

    def block(xx):
        return fn(xx)

    # non-reentrant checkpoint: fwd under no_grad, recompute under grad
    with torch.no_grad():
        # forward pass (no_grad) — capture saved shapes
        pass
    REGION['cur'] = 'fwd'
    out = ckpt.checkpoint(block, x, use_reentrant=False)
    REGION['cur'] = None
    loss = out.sum()

    # capture recompute saves via saved_tensors_hooks during backward
    region = {'re': False}
    def rpack(t):
        if region['re']:
            RECOMP_SHAPES.append(tuple(t.shape))
        return t
    def runpack(h):
        return h
    handle = torch.autograd.graph.saved_tensors_hooks(rpack, runpack)
    try:
        with handle:
            region['re'] = True
            loss.backward()
            torch.cuda.synchronize()
            ok = True
    except Exception as e:
        ok = False
        err = str(e).splitlines()[0]
    finally:
        region['re'] = False
        del handle
    return ok, FWD_SHAPES, RECOMP_SHAPES


for kind in ['F0', 'F1', 'F2', 'F3', 'F4']:
    ok, fwd, rec = run_case(kind)
    # focus on the div-relevant shapes: the [B,H,T,1,1] and [B,H,T,D,2] pair
    fwd_pair = [s for s in fwd if s in ((B, H, T, 1, 1), (B, H, T, D, 2))]
    rec_pair = [s for s in rec if s in ((B, H, T, 1, 1), (B, H, T, D, 2))]
    status = 'PASS' if ok else 'CRASH'
    match = 'ORDER-STABLE' if fwd_pair == rec_pair else f'ORDER-SWAP fwd={fwd_pair} rec={rec_pair}'
    print(f"{kind}: {status:6s}  {match}")
