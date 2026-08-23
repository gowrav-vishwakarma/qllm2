#!/usr/bin/env python3
"""Pinpoint the checkpoint saved-vs-recomputed mismatch with FULL context.

Two independent probes, both on the tiny 1-block model (math breaks identically
at any scale):

1. SAVED-STREAM PROBE: torch.utils.checkpoint.saved_tensors_hooks fires on
   EVERY tensor saved for backward (during forward) and every tensor
   recomputed (during backward). We log (idx, shape, dtype, requires_grad,
   is_grad_enabled, is_inference, value-checksum-for-small-tensors) for both
   streams and print a side-by-side diff at the FIRST mismatch — this is the
   "ping when the stage doesn't match" you asked for.

2. GRAD-FLOW PROBE: after backward, per-block parameter gradient norms.
   Verifies the old "only the last two layers received gradient" observation
   directly on this model.

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_probe [n_layers]
"""
from __future__ import annotations
import sys
import torch
import torch.utils.checkpoint as ckpt

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13Config, V13LM

N_LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 1

cfg = V13Config(vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=N_LAYERS,
    expand=2, dropout=0.0, max_seq_len=256, chunk_size=64,
    gradient_checkpointing=True, n_states=3, state_dt_spread=2.0,
    write_mode='delta', delta_chunk=32, delta_erase_gate=True,
    gate_content_aware=True, vault_state=True, vault_state_idx=0,
    write_phase_address=True, fused_e3=True, gate_surprisal_lambda=0.1,
    delta_key_norm=True, delta_erase_beta_cap=0.95)
torch.manual_seed(0)
m = V13LM(cfg).cuda()
m.train()
B, T = 2, 128
ids = torch.randint(0, 50257, (B, T), device='cuda')
lab = torch.randint(0, 50257, (B, T), device='cuda')

# ── Probe 1: saved-tensor stream with context ────────────────────────────────
SAVED = []    # (idx, shape, dtype, req_grad, grad_enabled, in inference, checksum)
RECOMP = []
CKSUM_MAX_ELEMS = 1 << 20  # only checksum small tensors (this model is tiny anyway)


def _ctx(t):
    import torch.autograd.profiler as prof
    return (
        tuple(t.shape), t.dtype, bool(t.requires_grad),
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
    )


def _cksum(t):
    try:
        if t.numel() <= CKSUM_MAX_ELEMS and t.is_floating_point():
            v = t.detach().float().to('cpu')
            return (v.mean().item(), v.abs().mean().item(), v.norm().item())
    except Exception:
        pass
    return None


def pack(x):
    SAVED.append(_ctx(x) + (_cksum(x),))
    return x


def unpack(x):
    RECOMP.append(_ctx(x) + (_cksum(x),))
    return x


def sig(e):
    return e[:5]  # shape, dtype, req_grad, grad_enabled, inference


try:
    handle = torch.autograd.graph.saved_tensors_hooks(pack, unpack)
    with handle:
        lm, _, _ = m._hidden_to_lm(ids)
        loss = m.ce_from_lm(lm, lab, chunk=4096)
        loss.backward()
    torch.cuda.synchronize()
    CRASH = None
except Exception as e:
    CRASH = str(e).splitlines()[0]
    print(f"CRASH: {CRASH}\n")

n = min(len(SAVED), len(RECOMP))
print(f"saved stream: {len(SAVED)} tensors | recompute stream: {len(RECOMP)} tensors | compared: {n}")
mismatches = [i for i in range(n) if sig(SAVED[i]) != sig(RECOMP[i])]
print(f"metadata mismatches: {len(mismatches)}  at positions {mismatches[:10]}{' ...' if len(mismatches)>10 else ''}")
for i in mismatches[:6]:
    s, r = SAVED[i], RECOMP[i]
    print(f"\n  pos {i}:")
    print(f"    saved  shape={s[0]} dtype={s[1]} req_grad={s[2]} grad_en={s[3]} inf_mode={s[4]} cksum={s[5]}")
    print(f"    recomp shape={r[0]} dtype={r[1]} req_grad={r[2]} grad_en={r[3]} inf_mode={r[4]} cksum={r[5]}")
# neighbors for context
if mismatches:
    i = mismatches[0]
    print(f"\n  context around first mismatch pos {i}:")
    for j in range(max(0, i - 3), min(n, i + 4)):
        s, r = SAVED[j], RECOMP[j]
        mark = ">>" if j in mismatches else "  "
        same = "same" if sig(s) == sig(r) else "DIFF"
        print(f"  {mark} pos {j:3d} [{same:4s}] saved shape={s[0]} cksum={s[5] and s[5][1]}  |  recomp shape={r[0]} cksum={r[5] and r[5][1]}")

# ── Probe 2: per-layer gradient flow ─────────────────────────────────────────
print("\n" + "=" * 70)
print("GRAD-FLOW PROBE: per-block param grad L2 norm after backward")
if CRASH is None:
    for i, block in enumerate(m.blocks):
        norms = []
        for pname, p in block.named_parameters(recurse=True):
            if p.grad is not None:
                norms.append(float(p.grad.norm()))
            else:
                norms.append(float('nan'))
        total = sum(x for x in norms if x == x)
        nz = sum(1 for x in norms if x == x and x > 0)
        print(f"  block {i}: params={len(norms)} nonzero_grad={nz} grad_norm_sum={total:.6f}")
else:
    print("  (skipped — backward crashed)")
