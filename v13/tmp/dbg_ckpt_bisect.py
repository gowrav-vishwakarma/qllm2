#!/usr/bin/env python3
"""Bisect the checkpoint recompute mismatch to a specific config feature.

Same 1-block repro as dbg_ckpt_mismatch, but each run flips ONE feature off
(relative to the production preset). If a feature's absence makes the crash
disappear, that feature contains the grad-mode shape divergence.

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_bisect
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13Config, V13LM


def base_cfg(**over) -> V13Config:
    cfg = dict(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=1,
        expand=2, dropout=0.0, max_seq_len=256, chunk_size=64,
        gradient_checkpointing=True, n_states=3, state_dt_spread=2.0,
        write_mode='delta', delta_chunk=32, delta_erase_gate=True,
        gate_content_aware=True, vault_state=True, vault_state_idx=0,
        write_phase_address=True, fused_e3=True, gate_surprisal_lambda=0.1,
        delta_key_norm=True, delta_erase_beta_cap=0.95,
    )
    cfg.update(over)
    return V13Config(**cfg)


def trial(**over) -> str:
    torch.manual_seed(0)
    m = V13LM(base_cfg(**over)).cuda()
    m.train()
    B, T = 2, 128
    ids = torch.randint(0, 50257, (B, T), device='cuda')
    lab = torch.randint(0, 50257, (B, T), device='cuda')
    try:
        lm, _, _ = m._hidden_to_lm(ids)
        loss = m.ce_from_lm(lm, lab, chunk=4096)
        loss.backward()
        torch.cuda.synchronize()
        return 'PASS'
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        return f'CRASH ({msg[:90]})'
    finally:
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    print(f"{'baseline':30s} {trial()}", flush=True)
    for off in [
        {'write_phase_address': False},
        {'delta_key_norm': False},
        {'use_rope': False},
        {'fused_qkv': False},
        {'qk_norm': True},
        {'gate_content_aware': False},
        {'gate_surprisal_lambda': 0.0},
        {'delta_erase_gate': False},
        {'vault_state': False},
        {'n_states': 1},
        {'decay_mode': 'per_channel'},
        {'fused_e3': False},
    ]:
        torch.cuda.empty_cache()
        print(f"{str(off):30s} {trial(**off)}", flush=True)
