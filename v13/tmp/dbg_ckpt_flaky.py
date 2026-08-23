#!/usr/bin/env python3
"""Is the checkpoint mismatch flaky (non-deterministic)?

Run the ORIGINAL code (delta_key_norm=True, all features on) N fresh
forward+backward passes and count crashes. If it's deterministic, every run
crashes or none do. If flaky, we see a mix.

Also re-runs the bisection toggles (delta_key_norm=False, write_phase_address=False,
gate_content_aware=False) N times each to check whether the earlier PASS
conclusions hold or were luck.

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_flaky [n]
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')
import v13.model as M
from v13.model import V13Config, V13LM

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def base_cfg(**over):
    d = dict(vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=1,
        expand=2, dropout=0.0, max_seq_len=256, chunk_size=64,
        gradient_checkpointing=True, n_states=3, state_dt_spread=2.0,
        write_mode='delta', delta_chunk=32, delta_erase_gate=True,
        gate_content_aware=True, vault_state=True, vault_state_idx=0,
        write_phase_address=True, fused_e3=True, gate_surprisal_lambda=0.1,
        delta_key_norm=True, delta_erase_beta_cap=0.95)
    d.update(over)
    return V13Config(**d)


def run_once(cfg):
    torch.manual_seed(0)
    m = V13LM(cfg).cuda()
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
        return f'CRASH({type(e).__name__})'
    finally:
        del m
        torch.cuda.empty_cache()


CASES = {
    'original (all on)': {},
    'delta_key_norm=False': dict(delta_key_norm=False),
    'write_phase_address=False': dict(write_phase_address=False),
    'gate_content_aware=False': dict(gate_content_aware=False),
    'fused_qkv=False': dict(fused_qkv=False),
}

for name, over in CASES.items():
    results = [run_once(base_cfg(**over)) for _ in range(N)]
    crashes = sum('CRASH' in r for r in results)
    print(f"{name:28s} {crashes}/{N} crashes   {' '.join(results)}", flush=True)
