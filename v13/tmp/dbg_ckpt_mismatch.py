#!/usr/bin/env python3
"""Localize the non-reentrant checkpoint recompute mismatch to a specific op.

1-block model, checkpoint debug ON, forward+backward. Prints the full
CheckpointError (torch dumps every op of both passes); grep the output for the
mismatched shapes to name the offending op.

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_mismatch > /tmp/ckpt_dbg.txt 2>&1
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13Config, V13LM

B, T = 2, 128
cfg = V13Config(vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=1,
    expand=2, dropout=0.0, max_seq_len=256, chunk_size=64, gradient_checkpointing=True,
    n_states=3, state_dt_spread=2.0, write_mode='delta', delta_chunk=32,
    delta_erase_gate=True, gate_content_aware=True, vault_state=True, vault_state_idx=0,
    write_phase_address=True, fused_e3=True, gate_surprisal_lambda=0.1,
    delta_key_norm=True, delta_erase_beta_cap=0.95)
torch.manual_seed(0)
m = V13LM(cfg).cuda()
m.train()
ids = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')
lab = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')

with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):
    lm, _, _ = m._hidden_to_lm(ids)
    loss = m.ce_from_lm(lm, lab, chunk=4096)
    try:
        loss.backward()
        print("NO CRASH (mismatch gone)")
    except Exception as e:
        print("CRASH:", type(e).__name__)
        print(str(e))
