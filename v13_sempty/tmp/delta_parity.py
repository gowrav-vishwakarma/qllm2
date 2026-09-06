"""Parity: pam_delta_batched vs pam_delta_torch (forward read/state and all grads).

Usage:  PYTHONPATH=. .venv/bin/python v13_sempty/tmp/delta_parity.py
"""
import sys
import torch

sys.path.insert(0, '.')
from v13_sempty.triton_kernels import pam_delta_torch, pam_delta_batched  # noqa: E402


def case(BH, T, K, dtype, carry_in, seed=0, chunk=256):
    torch.manual_seed(seed)
    dev = 'cuda'
    q = torch.randn(BH, T, K, device=dev)
    k = torch.nn.functional.normalize(torch.randn(BH, T, K, device=dev), dim=-1)
    v = torch.randn(BH, T, K, device=dev)
    ret = torch.rand(BH, T, device=dev) * 0.5 + 0.5
    bw = torch.rand(BH, T, device=dev)
    be = torch.rand(BH, T, device=dev) * 0.95
    carry = torch.randn(BH, K, K, device=dev) * 0.1 if carry_in else None
    leaves = {}
    outs = {}
    for name, fn in (('ref', pam_delta_torch), ('new', pam_delta_batched)):
        ts = [t.clone().to(dtype if t.dim() == 3 else torch.float32).requires_grad_(True)
              for t in (q, k, v)]
        rs = [t.clone().requires_grad_(True) for t in (ret, bw, be)]
        c = carry.clone().requires_grad_(True) if carry is not None else None
        if name == 'ref':
            read, S = fn(*ts, *rs, c, chunk)
        else:
            read, S = fn(*ts, *rs, c)
        loss = (read.float() * torch.linspace(0.5, 1.5, K, device=dev)).sum() + (S * 0.3).sum()
        loss.backward()
        grads = [t.grad for t in ts + rs] + ([c.grad] if c is not None else [])
        outs[name] = (read.float(), S.float(), [g.float() for g in grads])
    names = ['dq', 'dk', 'dv', 'dret', 'dbw', 'dbe'] + (['dcarry'] if carry_in else [])
    r0, s0, g0 = outs['ref']
    r1, s1, g1 = outs['new']

    def rel(a, b):
        return ((a - b).norm() / (a.norm() + 1e-12)).item()

    rows = [('read', rel(r0, r1)), ('state', rel(s0, s1))] + [(n, rel(a, b)) for n, a, b in zip(names, g0, g1)]
    worst = max(x for _, x in rows)
    tol = 2e-4 if dtype == torch.float32 else 3e-2
    ok = worst <= tol
    print(f"BH={BH} T={T} K={K} {str(dtype).split('.')[-1]:8s} carry={carry_in!s:5s} "
          f"worst_rel={worst:.2e} tol={tol:.0e} {'OK' if ok else 'FAIL'}   "
          + ' '.join(f"{n}={x:.1e}" for n, x in rows))
    return ok


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    all_ok = True
    for dtype in (torch.float32, torch.bfloat16):
        for (BH, T, K, ci) in ((4, 512, 98, False), (4, 512, 98, True), (3, 300, 64, True),
                               (2, 2048, 98, False), (6, 64, 98, True)):
            all_ok &= case(BH, T, K, dtype, ci)
    print('ALL OK' if all_ok else 'SOME FAILED')
    sys.exit(0 if all_ok else 1)
