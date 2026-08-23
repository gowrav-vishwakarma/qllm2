#!/usr/bin/env python3
"""Verify V13 block gradient flow (regression test for the _ckpt_block bug).

1) With gradient_checkpointing=True, EVERY block must receive a nonzero main-loss
   gradient (the d0abeed detach froze all but the last).
2) The checkpointed backward must match the non-checkpointed backward (same grads
   up to tolerance), proving the fix is math-exact, not just "unfrozen".

Usage: .venv/bin/python -m v13.tmp.test_gradflow
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.model import V13Config, V13LM


def make_cfg(ckpt: bool) -> V13Config:
    return V13Config(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=4,
        expand=2, dropout=0.0, max_seq_len=256, chunk_size=64,
        gradient_checkpointing=ckpt, n_states=3, state_dt_spread=2.0,
        write_mode='delta', delta_chunk=32, delta_erase_gate=True,
        gate_content_aware=True, vault_state=True, vault_state_idx=0,
        write_phase_address=True, fused_e3=True, gate_surprisal_lambda=0.1,
        delta_key_norm=True, delta_erase_beta_cap=0.95,
    )


def run_backward(cfg: V13Config):
    model = V13LM(cfg).cuda()
    model.train()
    B, T = 2, 128
    ids = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')
    labels = torch.randint(0, cfg.vocab_size, (B, T), device='cuda')
    lm, _aux, _gp = model._hidden_to_lm(ids)
    loss = model.ce_from_lm(lm, labels, chunk=4096)
    loss.backward()
    return model, float(loss.detach())


def grad_of(model, i, kind):
    if kind == 'qkv':
        return model.blocks[i].pam.qkv_proj.weight_real.grad
    if kind == 'cgu':
        return model.blocks[i].cgu.up_proj.weight_real.grad
    if kind == 'dtb':
        return model.blocks[i].pam.dt_bias.grad
    raise ValueError(kind)


def main():
    torch.manual_seed(0)
    print("=== [1] all-blocks-train (gradient_checkpointing=True) ===")
    m_ckpt, loss = run_backward(make_cfg(ckpt=True))
    print(f"loss={loss:.4f}")
    frozen = []
    for i in range(4):
        g = grad_of(m_ckpt, i, 'qkv')
        gn = None if g is None else float(g.norm())
        print(f"  block {i} qkv grad norm = {gn}")
        if gn is None or gn == 0.0:
            frozen.append(i)
    ok1 = not frozen
    print("  -> ALL BLOCKS TRAIN" if ok1 else f"  -> BUG: frozen blocks {frozen}")

    print("\n=== [2] checkpointed vs non-checkpointed gradient equality ===")
    torch.manual_seed(123)
    m_ref, _ = run_backward(make_cfg(ckpt=False))
    torch.manual_seed(123)
    m_chk, _ = run_backward(make_cfg(ckpt=True))
    max_rel = 0.0
    for i in range(4):
        for kind in ('qkv', 'cgu', 'dtb'):
            gr = grad_of(m_ref, i, kind)
            gc = grad_of(m_chk, i, kind)
            denom = max(float(gr.norm()), 1e-8)
            max_rel = max(max_rel, float((gr - gc).norm()) / denom)
    print(f"  max relative grad mismatch (ckpt vs no-ckpt) = {max_rel:.3e}")
    ok2 = max_rel < 1e-4
    print("  -> MATCH" if ok2 else "  -> MISMATCH (checkpoint not math-exact)")

    print("\n" + ("PASS" if (ok1 and ok2) else "FAIL"))
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == '__main__':
    main()
