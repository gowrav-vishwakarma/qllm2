#!/usr/bin/env python3
"""VERIFICATION ONLY. Confirms the other LLM's claim that the SCRATCHPAD's
proposed autograd.Function backward `return g / mag` is mathematically WRONG.

cnormalize_vec(x) = x / m where m = sqrt(sum(x^2) + eps) (m a scalar per
vector, depends on all x). The true Jacobian is (I - x_hat x_hat^T)/m, so
the true gradient is gx = g/m - x*(g.x)/m^3. The proposed `g/m` DROPS the
second term x*(g.x)/m^3.

This compares, on random data:
  TRUE   = autograd gradient of the plain (unscripted) x/m
  PROPOSED = the SCRATCHPAD Function's g/m
and reports the relative L2 error. If ~0.1 (not ~0), the proposed fix would
corrupt key gradients.
"""
from __future__ import annotations
import torch
sys = __import__('sys')
sys.path.insert(0, '/home/gowrav/Development/qllm2')

torch.manual_seed(0)
B, H, T, D = 2, 2, 128, 32
x = torch.randn(B, H, T, D, 2, requires_grad=True)
g = torch.randn(B, H, T, D, 2)  # upstream grad dL/dy


def true_grad(x):
    mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
    out = x / mag.unsqueeze(-1).unsqueeze(-1)
    # backprop a fixed upstream gradient g
    out.backward(g, retain_graph=True)
    grad = x.grad.clone()
    x.grad = None
    return grad


def proposed_grad(x):
    # The SCRATCHPAD Function: forward saves only mag, backward returns g/mag.
    mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
    return g / mag.unsqueeze(-1).unsqueeze(-1)


gt = true_grad(x)
gp = proposed_grad(x)
rel = (gt - gp).norm() / gt.norm()
# also the analytic correct form for a cross-check
mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
m3 = mag ** 3
gx_correct = g / mag.unsqueeze(-1).unsqueeze(-1) - x * (
    (g * x).sum((-2, -1)).unsqueeze(-1).unsqueeze(-1)) / m3.unsqueeze(-1).unsqueeze(-1)
rel_correct = (gt - gx_correct).norm() / gt.norm()
print(f"TRUE autograd grad norm:        {gt.norm():.6e}")
print(f"PROPOSED (g/mag) grad norm:     {gp.norm():.6e}")
print(f"rel L2 error PROPOSED vs TRUE:  {rel:.6e}   <-- if ~0.1, the fix is WRONG")
print(f"rel L2 error analytic vs TRUE:  {rel_correct:.6e}   <-- should be ~1e-16 (validates method)")
