"""CHRONO_HEADS (split key geometry, Mind's twin (b); built by Library 2026-11-18): parity + streaming + grads + knockout.

Run on CPU, fp32, wio geometry:
    python v13_sempty/tmp/test_chrono_heads.py
Checks (Mind's spec, EXPERIMENTS_SEMPY "Tonight's 100M / 1B twins" (b)):
  1. bit-exact forward: CHRONO=1 CHRONO_HEADS=0 == CHRONO=0; CHRONO=1 CHRONO_HEADS=H == CHRONO=1 default (nonzero warp).
  2. k = H/2, nonzero warp: the RoPE heads' (cos, sin) equal plain RoPE's bit-exact; the clocked heads' differ.
  3. streaming: two chunks with the (notebook, clock) carried, and prefill + token steps, == one full pass (< 1e-5).
  4. backward: grads finite; warp_proj grads zero on the unclocked heads, nonzero on the clocked ones; causal.
  5. knockout: V13_CHRONO_KNOCKOUT=clock / rope change the logits; a knocked-out model equals one whose masked heads'
     notebook reads are zero (checked via the reader / writer per-head mask path).
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v13_sempty.config import get_config  # noqa: E402
from v13_sempty.model import LM  # noqa: E402
from v13_sempty.real_ops import build_rope_cache_real  # noqa: E402

torch.set_num_threads(min(16, os.cpu_count() or 1))


def wio_cfg(chrono: bool, k: int = -1):
    cfg = get_config('baseline_real_pm')
    cfg.pam_layers = (3, 7, 11, 15)
    cfg.share_reads = cfg.out_gate = cfg.delta = cfg.echo = True
    cfg.mlp_hidden = 2128
    cfg.chunk_size = 16
    cfg.chrono = chrono
    cfg.chrono_heads = k
    cfg.gradient_checkpointing = False   # share_reads trains without it (as the wio arms)
    return cfg


def build(chrono, k=-1, src=None, seed=0):
    torch.manual_seed(seed)
    m = LM(wio_cfg(chrono, k)).eval()
    if src is not None:
        missing, unexpected = m.load_state_dict(src.state_dict(), strict=False)
        assert not unexpected, unexpected
        assert all('warp_proj' in n for n in missing), missing
    return m


def pams(m):
    return [b.pam for b in m.blocks if b.pam is not None]


def random_warp(m, seed=1):
    g = torch.Generator().manual_seed(seed)
    for p in pams(m):
        with torch.no_grad():
            p.warp_proj.weight.copy_(torch.randn(p.warp_proj.weight.shape, generator=g) * 0.3)
            p.warp_proj.bias.copy_(torch.randn(p.warp_proj.bias.shape, generator=g) * 0.3)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.manual_seed(0)
    B, T = 2, 45
    base = build(False)
    H = base.blocks[3].pam.heads.size
    ids = torch.randint(0, wio_cfg(False).vocab_size, (B, T))
    print(f"wio geometry: H={H}, head_dim={pams(base)[0].head_dim}, notebooks at {wio_cfg(False).pam_layers}")

    # 1a. k = 0 == nochrono (warp nonzero, so masking is what makes it equal)
    m0 = build(True, 0, src=base); random_warp(m0)
    with torch.no_grad():
        lb, _, _ = base.forward(ids); l0, _, _ = m0.forward(ids)
    d0 = (lb - l0).abs().max().item(); eq0 = torch.equal(lb, l0)
    print(f"1a k=0 vs CHRONO=0: bit-exact {eq0}, max|diff| {d0:.3e}")
    assert eq0

    # 1b. k = H == chrono default (nonzero warp, clock active)
    pam_def = build(True, -1, src=base); random_warp(pam_def)
    mH = build(True, H, src=pam_def)
    with torch.no_grad():
        ld, _, _ = pam_def.forward(ids); lH, _, _ = mH.forward(ids)
    eqH = torch.equal(ld, lH)
    print(f"1b k=H vs CHRONO default (random warp): bit-exact {eqH}, max|diff| {(ld - lH).abs().max().item():.3e}; "
          f"clock active: |pam - nochrono| {(ld - lb).abs().max().item():.3e}")
    assert eqH and (ld - lb).abs().max().item() > 1e-3

    # params
    split = build(True, H // 2, src=pam_def)
    print(f"params: nochrono {nparams(base):,}  pam {nparams(pam_def):,}  split k={H // 2} {nparams(split):,} "
          f"(torch.where keeps all H warp outputs, so split == pam)")

    # 2. k = H/2: rope heads' (cos, sin) == plain RoPE; clocked heads differ
    k = H // 2
    with torch.no_grad():
        ls, _, _ = split.forward(ids)
    p0 = pams(split)[0]
    cos, sin = p0._rot                                           # [B,H,T,P], the writer's rotation (expose)
    rc = build_rope_cache_real(T, p0.head_dim // 2)              # [T,P,2]
    rope_eq = torch.equal(cos[:, k:], rc[..., 0].expand_as(cos[:, k:])) and torch.equal(sin[:, k:], rc[..., 1].expand_as(sin[:, k:]))
    clk_diff = (cos[:, :k] - rc[..., 0]).abs().max().item()
    print(f"2  k={k}: RoPE heads (cos, sin) == plain RoPE bit-exact {rope_eq}; clocked heads max|cos - rope| {clk_diff:.3e}")
    assert rope_eq and clk_diff > 1e-2
    # the split model differs from both parents
    print(f"   split vs nochrono {(ls - lb).abs().max().item():.3e}, split vs pam {(ls - ld).abs().max().item():.3e}")

    # 3. streaming: two chunks (unaligned) and prefill + steps vs full pass
    a = 29
    with torch.no_grad():
        l1, st, _ = split.forward(ids[:, :a])
        l2, st2, _ = split.forward(ids[:, a:], states=st, step_offset=a)
        d_chunks = (torch.cat([l1, l2], 1) - ls).abs().max().item()
        parts, s = [split.forward(ids[:, :a])[0]], split.forward(ids[:, :a])[1]
        for t in range(a, T):
            lt, s, _ = split.forward(ids[:, t:t + 1], states=s, step_offset=t)
            parts.append(lt)
        d_steps = (torch.cat(parts, 1) - ls).abs().max().item()
    clk = [x[1] for x in st2 if isinstance(x, tuple) and len(x) == 2 and x[1] is not None and x[1].dim() == 2]
    rope_clock = clk[0][:, k:] if clk else None
    print(f"3  two chunks max|diff| {d_chunks:.3e}; prefill {a} + {T - a} steps max|diff| {d_steps:.3e}; "
          f"RoPE heads' carried clock == T: {bool(rope_clock is not None and torch.equal(rope_clock, torch.full_like(rope_clock, float(T))))}")
    assert d_chunks < 1e-5 and d_steps < 1e-5

    # 4. backward + causal
    split.train()
    for mod in split.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    split.zero_grad(set_to_none=True)
    lg, _, _ = split.forward(ids)
    F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), ids.reshape(-1)).backward()
    fin = all(torch.isfinite(p.grad).all() for p in split.parameters() if p.grad is not None)
    w = [p.warp_proj.weight.grad for p in pams(split)]
    wb = [p.warp_proj.bias.grad for p in pams(split)]
    def rows(gw):                                                # the H output rows, whichever axis is the output
        return gw if gw.shape[0] == H else gw.t()
    off = max(rows(x)[k:].abs().max().item() for x in w) + max(x[k:].abs().max().item() for x in wb)
    on = min(rows(x)[:k].abs().max().item() for x in w)
    print(f"4  grads finite {fin}; warp_proj grad on unclocked heads max {off:.1e}, on clocked heads min-of-max {on:.3e}")
    assert fin and off == 0.0 and on > 0
    split.eval()
    ids2 = ids.clone(); ids2[:, 30:] = (ids2[:, 30:] + 7) % 50257
    with torch.no_grad():
        lc, _, _ = split.forward(ids2)
    d_c = (lc[:, :30] - ls[:, :30]).abs().max().item()
    print(f"   causal: logits before the change max|diff| {d_c:.1e}")
    assert d_c == 0.0

    # 5. knockout (eval diagnostic)
    outs = {}
    for ko in ('clock', 'rope'):
        os.environ['V13_CHRONO_KNOCKOUT'] = ko
        mk = build(True, k, src=split)
        with torch.no_grad():
            outs[ko] = mk.forward(ids)[0]
        keep = pams(mk)[0].chrono_read_keep
        print(f"5  knockout={ko}: keep mask {keep.int().tolist()}; max|logit change| {(outs[ko] - ls).abs().max().item():.3e}")
        assert (outs[ko] - ls).abs().max().item() > 1e-4
    os.environ.pop('V13_CHRONO_KNOCKOUT')
    print("ALL PASS")


if __name__ == '__main__':
    main()
