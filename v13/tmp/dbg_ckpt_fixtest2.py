#!/usr/bin/env python3
"""High-repetition flaky+fix test.

The checkpoint mismatch is FLAKY (a 2-tensor order-swap of the cnormalize_vec
div operands, detected by the determinism check). Single-run tests are
meaningless. This runs each candidate formulation N fresh forward+backward
passes and reports the crash COUNT.

Candidates:
  ORIG  current cnormalize_vec (x / mag.unsqueeze(-1).unsqueeze(-1))
  RECIP x * mag_uu.reciprocal()
  RSQRT x * rsqrt(norm_sq).uu
  FUNC  explicit torch.autograd.Function (fixed save order)

A robust fix: 0 crashes in N runs while ORIG crashes >0.

Usage: .venv/bin/python -m v13.tmp.dbg_ckpt_fixtest2 [n]
"""
from __future__ import annotations
import sys
import torch

sys.path.insert(0, '/home/gowrav/Development/qllm2')
import v13.model as M
from v13.model import V13Config, V13LM

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
_orig_cnvec = M.cnormalize_vec


def C_orig(x):
    mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
    return x / mag.unsqueeze(-1).unsqueeze(-1)


def C_recip(x):
    mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
    return x * mag.unsqueeze(-1).unsqueeze(-1).reciprocal()


def C_rsqrt(x):
    norm_sq = (x[..., 0].square() + x[..., 1].square()).sum(-1)
    inv = torch.rsqrt(norm_sq + 1e-8)
    return x * inv.unsqueeze(-1).unsqueeze(-1)


class _CFunc(torch.autograd.Function):
    """Checkpoint-safe cnormalize_vec.

    out = x / mag, where mag = sqrt(sum(x^2, -1) + eps).
    Saves ONLY mag (a single tensor) — there is no 2-operand div whose save
    order can swap between the checkpoint forward and the recompute.
    """
    @staticmethod
    def forward(ctx, x):
        mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
        out = x / mag.unsqueeze(-1).unsqueeze(-1)
        ctx.save_for_backward(mag)
        return out

    @staticmethod
    def backward(ctx, g):
        (mag,) = ctx.saved_tensors
        # d(out)/d(x) = 1 / mag ; mag has no independent gradient path.
        return g / mag.unsqueeze(-1).unsqueeze(-1)

def C_func(x):
    return _CFunc.apply(x)


CANDIDATES = {
    'ORIG (x / mag_uu)': C_orig,
    'RECIP (x * mag_uu.recip)': C_recip,
    'RSQRT (x * rsqrt.uu)': C_rsqrt,
    'FUNC (autograd.Function)': C_func,
}


def run_once(fn):
    M.cnormalize_vec = fn
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
    try:
        lm, _, _ = m._hidden_to_lm(ids)
        loss = m.ce_from_lm(lm, lab, chunk=4096)
        loss.backward()
        torch.cuda.synchronize()
        return 'PASS'
    except Exception as e:
        return f'CRASH'
    finally:
        M.cnormalize_vec = _orig_cnvec
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    for name, fn in CANDIDATES.items():
        crashes = 0
        for _ in range(N):
            if 'CRASH' in run_once(fn):
                crashes += 1
        print(f"{name:26s} {crashes}/{N} crashes", flush=True)
