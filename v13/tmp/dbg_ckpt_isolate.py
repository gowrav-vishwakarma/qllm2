#!/usr/bin/env python3
"""Localize the checkpoint recompute mismatch to _project vs the fused delta step.

Runs each PAM sub-path through grad_checkpoint separately (original forward
under no_grad, recompute under grad) and checks whether backward crashes:

  A. _project only          (qkv/rope/phase/delta-key-norm)
  B. _gamma_and_vprime only (decay + GSP protect gate)
  C. full PAM.forward       (everything, incl. fused delta solve)

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_isolate
"""
from __future__ import annotations
import sys
import torch
import torch.utils.checkpoint as ckpt

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13Config, V13PAMLayer

cfg = V13Config(vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=1,
    expand=2, dropout=0.0, max_seq_len=256, chunk_size=64,
    n_states=3, state_dt_spread=2.0, write_mode='delta', delta_chunk=32,
    delta_erase_gate=True, gate_content_aware=True, vault_state=True,
    vault_state_idx=0, write_phase_address=True, fused_e3=True,
    gate_surprisal_lambda=0.1, delta_key_norm=True, delta_erase_beta_cap=0.95)
torch.manual_seed(0)
pam = V13PAMLayer(cfg).cuda()
pam.train()
B, T = 2, 128
# PAM input layout: [B, T, dim, 2] (complex split-real)
x = torch.randn(B, T, cfg.dim, 2, device='cuda', requires_grad=True)


def run(name, fn):
    try:
        out = ckpt.checkpoint(fn, x, use_reentrant=False)
        if not isinstance(out, torch.Tensor):
            out = out[0]
        out.sum().backward()
        torch.cuda.synchronize()
        print(f"{name:24s} PASS")
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"{name:24s} CRASH ({msg[:90]})")
    finally:
        pam.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.empty_cache()


run('A _project', lambda t: pam._project(t, 0)[0])
run('B gamma_vprime', lambda t: pam._gamma_and_vprime(t, pam._project(t, 0)[2])[0])
run('C full pam', lambda t: pam(t, state=None, step_offset=0)[0])
