#!/usr/bin/env python3
"""Test cnormalize_vec fix formulations on the REAL 1-block model.

Monkeypatches v13.model.cnormalize_vec (the name bound at the call site
model.py:391) with each candidate formulation, runs forward+backward, and
reports PASS/CRASH. The correct fix makes the checkpoint backward pass.

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_fixtest
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')
import v13.model as M
from v13.model import V13Config, V13LM

_orig_cnvec = M.cnormalize_vec


def C0_current(x):
    mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
    return x / mag.unsqueeze(-1).unsqueeze(-1)


def C1_reciprocal(x):
    mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
    return x * mag.unsqueeze(-1).unsqueeze(-1).reciprocal()


def C2_rsqrt(x):
    norm_sq = (x[..., 0].square() + x[..., 1].square()).sum(-1)
    inv = torch.rsqrt(norm_sq + 1e-8)
    return x * inv.unsqueeze(-1).unsqueeze(-1)


def C3_contig(x):
    mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
    return x / mag.unsqueeze(-1).unsqueeze(-1).contiguous()


def C4_explicit_div(x):
    mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
    return torch.div(x, mag.unsqueeze(-1).unsqueeze(-1))


def C5_rsqrt_contig(x):
    norm_sq = (x[..., 0].square() + x[..., 1].square()).sum(-1)
    inv = torch.rsqrt(norm_sq + 1e-8)
    return x * inv.unsqueeze(-1).unsqueeze(-1).contiguous()


CANDIDATES = {
    'C0 current (x / mag_uu)': C0_current,
    'C1 x * mag_uu.reciprocal()': C1_reciprocal,
    'C2 x * rsqrt(norm_sq).uu': C2_rsqrt,
    'C3 x / mag_uu.contiguous()': C3_contig,
    'C4 torch.div(x, mag_uu)': C4_explicit_div,
    'C5 x * rsqrt(norm_sq).uu.contig': C5_rsqrt_contig,
}


def build():
    torch.manual_seed(0)
    cfg = V13Config(vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=1,
        expand=2, dropout=0.0, max_seq_len=256, chunk_size=64,
        gradient_checkpointing=True, n_states=3, state_dt_spread=2.0,
        write_mode='delta', delta_chunk=32, delta_erase_gate=True,
        gate_content_aware=True, vault_state=True, vault_state_idx=0,
        write_phase_address=True, fused_e3=True, gate_surprisal_lambda=0.1,
        delta_key_norm=True, delta_erase_beta_cap=0.95)
    m = V13LM(cfg).cuda()
    m.train()
    B, T = 2, 128
    ids = torch.randint(0, 50257, (B, T), device='cuda')
    lab = torch.randint(0, 50257, (B, T), device='cuda')
    return m, ids, lab


def trial(name, fn):
    M.cnormalize_vec = fn
    m, ids, lab = build()
    try:
        lm, _, _ = m._hidden_to_lm(ids)
        loss = m.ce_from_lm(lm, lab, chunk=4096)
        loss.backward()
        torch.cuda.synchronize()
        return 'PASS'
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        return f'CRASH ({msg[:70]})'
    finally:
        M.cnormalize_vec = _orig_cnvec
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    for name, fn in CANDIDATES.items():
        torch.cuda.empty_cache()
        print(f"{name:36s} {trial(name, fn)}", flush=True)
