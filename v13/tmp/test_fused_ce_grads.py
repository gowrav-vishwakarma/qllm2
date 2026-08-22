"""fused_linear_cross_entropy gradient check vs torch reference.

r1 (Jul 1) used torch.nn.functional.cross_entropy (no fused_ce existed).
ab1 (now) uses --fused_ce. If the fused backward differs from the reference
in ANY way (mask handling, fp32 vs bf16 accumulation, ignore_index rows,
mean-vs-sum normalization), the optimizer trajectory diverges from step 1 —
exactly the observed symptom (step-0 loss identical, then monotonic drift).

Run: .venv/bin/python v13/tmp/test_fused_ce_grads.py
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/gowrav/Development/qllm2')
from v13.fused_ce import fused_linear_cross_entropy  # noqa: E402

torch.manual_seed(3)
dev = 'cuda'
N, V = 2048, 50261  # one B18/T2048 worth of rows
W = (torch.randn(V, 512, device=dev) * 0.02).to(torch.bfloat16)
H = torch.randn(N, 512, device=dev, dtype=torch.bfloat16)
tgt = torch.randint(0, V, (N,), device=dev)
mask = torch.ones(N, device=dev)
mask[:37] = 0.0           # some ignore rows (as in real training)
tgt[:37] = 0              # (real pipeline: labels=0 = ignore for pretrain? check)

def ref():
    H2 = H.clone().requires_grad_(True)
    W2 = W.clone().requires_grad_(True)
    logits = H2.float() @ W2.float().t()
    # standard masked mean CE
    ce = F.cross_entropy(logits, tgt, reduction='none')
    loss = (ce * mask).sum() / mask.sum()
    loss.backward()
    return loss.detach(), H2.grad.float(), W2.grad.float()

def fused():
    H3 = H.clone().requires_grad_(True)
    W3 = W.clone().requires_grad_(True)
    loss = fused_linear_cross_entropy(H3, W3, tgt, mask, chunk=4096)
    if isinstance(loss, tuple):
        loss = loss[0]
    loss.backward()
    return loss.detach(), H3.grad.float(), W3.grad.float()

l_ref, gh_ref, gw_ref = ref()
l_fus, gh_fus, gw_fus = fused()
print(f"ref loss  = {l_ref.item():.6f}")
print(f"fused loss= {l_fus.item():.6f}  (diff {abs(l_ref.item()-l_fus.item()):.3e})")
dh = (gh_ref - gh_fus).abs().max().item()
dw = (gw_ref - gw_fus).abs().max().item()
rh = (gh_ref - gh_fus).norm() / (gh_ref.norm() + 1e-12)
rw = (gw_ref - gw_fus).norm() / (gw_ref.norm() + 1e-12)
print(f"grad_hidden: max-abs {dh:.4e}  rel-L2 {rh:.4e}")
print(f"grad_weight: max-abs {dw:.4e}  rel-L2 {rw:.4e}")
print("VERDICT:", "EQUIVALENT" if (rh < 1e-4 and rw < 1e-4) else "DIVERGENT — fused CE backward is NOT the reference")
