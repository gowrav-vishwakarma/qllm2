"""Re-runnable: prove `_forward_multistate_delta_fused` (K-batched) == the
per-state K-loop `_forward_multistate` (delta) bit-for-bit, on forward output,
PAM state, aux, and gradients.

Run:
    .venv/bin/python v13/tmp/test_fused_delta_equivalence.py
    V13_DEBUG=1 .venv/bin/python v13/tmp/test_fused_delta_equivalence.py   # shape dump
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from v13.model import V13LM, V13Config


def cfg(fused: bool, ckpt: bool, **kw):
    base = dict(
        vocab_size=50261, dim=384, n_heads=6, head_dim=64, n_layers=2,
        expand=3, dropout=0.0, max_seq_len=2048, chunk_size=256,
        n_states=3, state_dt_spread=2.0, write_mode='delta', delta_chunk=64,
        delta_erase_gate=True, gate_content_aware=True, vault_state=True,
        vault_state_idx=0, write_phase_address=True, gate_surprisal_lambda=0.1,
        fused_e3=fused, gradient_checkpointing=ckpt,
    )
    base.update(kw)
    return V13Config(**base)


def main(B=2, T=512, tol=1e-4):
    torch.manual_seed(0)
    mA = V13LM(cfg(False, False)).cuda(); mA.eval()   # K-loop (ablation)
    mB = V13LM(cfg(True, False)).cuda();  mB.eval()   # fused-delta (new)
    mB.load_state_dict(mA.state_dict())
    ids = torch.randint(0, 50261, (B, T), device='cuda')

    with torch.no_grad():
        lA, auxA, _ = mA._hidden_to_lm(ids)
        lB, auxB, _ = mB._hidden_to_lm(ids)
    fwd_diff = (lA - lB).abs().max().item()
    aux_diff = abs(auxA.item() - auxB.item())
    print(f"forward:  max|logit diff| = {fwd_diff:.2e}")
    print(f"aux:      diff = {aux_diff:.2e}")

    h = mA.embed(ids)
    x0 = mA.blocks[0].norm1(h); x0 = x0 + mA.blocks[0].cgu(x0)
    inp = mA.blocks[0].norm2(x0)
    with torch.no_grad():
        oA, sA = mA.blocks[0].pam(inp, state=None)
        oB, sB = mB.blocks[0].pam(inp, state=None)
    out_diff = (oA - oB).abs().max().item()
    st_diff = (sA - sB).abs().max().item()
    print(f"pam out:  max|diff| = {out_diff:.2e}")
    print(f"state:    max|diff| = {st_diff:.2e}  shape={tuple(sA.shape)}")

    # Gradient equivalence (train mode, dropout 0).
    mA.train(); mB.train(); mA.zero_grad(); mB.zero_grad()
    lA2, _, _ = mA._hidden_to_lm(ids); mA.ce_from_lm(lA2, ids).backward()
    lB2, _, _ = mB._hidden_to_lm(ids); mB.ce_from_lm(lB2, ids).backward()
    gd = max((a.grad - b.grad).abs().max().item()
             for a, b in zip(mA.parameters(), mB.parameters())
             if a.grad is not None and b.grad is not None)
    print(f"grads:    max|diff| = {gd:.2e}")

    ok = fwd_diff < tol and st_diff < tol and gd < tol
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
