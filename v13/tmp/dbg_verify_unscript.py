#!/usr/bin/env python3
"""VERIFICATION ONLY (does not modify source). Confirms the other LLM's finding
that UNSCRIPTING cnormalize_vec (drop @torch.jit.script, body byte-identical)
fixes the checkpoint recompute mismatch, in a FRESH process.

Monkeypatches v13.model.cnormalize_vec (the name the call site model.py:391
resolves) with the SAME body but no decorator, then runs the 1-layer repro.
Also monkeypatches cabs to confirm unscripting cabs ALONE does NOT fix it
(specificity check: the culprit is cnormalize_vec, not any jit function).

Usage:
  .venv/bin/python -m v13.tmp.dbg_verify_unscript cnvec   # unscript cnormalize_vec (expect PASS)
  .venv/bin/python -m v13.tmp.dbg_verify_unscript cabs    # unscript cabs only (expect CRASH)
"""
from __future__ import annotations
import sys
import torch
sys.path.insert(0, '/home/gowrav/Development/qllm2')
import v13.model as M
from v13.model import V13Config, V13LM

WHICH = sys.argv[1] if len(sys.argv) > 1 else 'cnvec'


def cnvec_plain(x):
    mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
    return x / mag.unsqueeze(-1).unsqueeze(-1)


def cabs_plain(x):
    return torch.sqrt(x[..., 0].square() + x[..., 1].square() + 1e-8)


if WHICH == 'cnvec':
    M.cnormalize_vec = cnvec_plain
elif WHICH == 'cabs':
    M.cabs = cabs_plain
else:
    raise SystemExit("WHICH must be cnvec or cabs")

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
lm, _, _ = m._hidden_to_lm(ids)
loss = m.ce_from_lm(lm, lab, chunk=4096)
try:
    loss.backward()
    torch.cuda.synchronize()
    print(f"WHICH={WHICH}: NO CRASH")
except Exception as e:
    print(f"WHICH={WHICH}: CRASH {type(e).__name__}")
