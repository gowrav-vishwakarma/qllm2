"""Fused (chunked) linear + cross-entropy for the tied complex LM head.

The V11 head is algebraically a single real matmul:

    logits = lm_real @ E_real.T + lm_imag @ E_imag.T
           = concat(lm_real, lm_imag) @ concat(E_real, E_imag).T
           = H @ W.T                      H:[N,2d]  W:[V,2d]

For vocab V~50k and N=B*T~37k the logits tensor `[N, V]` (~4 GB fp32, plus an
equal-size softmax and grad) dominates training memory. This module computes the
mean/masked cross-entropy WITHOUT ever materializing the full `[N, V]` logits:

  * forward  processes H in row-chunks, keeping only `[chunk, V]` live, and saves
    just H, W, targets (not logits) for backward;
  * backward recomputes the per-chunk logits, forms the softmax gradient, and
    accumulates grad_H and grad_W chunk by chunk.

Peak head memory drops from O(N*V) to O(chunk*V). Math is exact (fp32 reduction),
verified against F.cross_entropy in `_test`.
"""

from typing import Optional

import torch
import torch.nn.functional as F


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, W, targets, mask, chunk, ignore_index):
        # H:[N,K] W:[V,K] targets:[N] mask:[N] or None
        N = H.shape[0]
        loss_sum = H.new_zeros(())
        # denominator = number of contributing tokens
        if mask is not None:
            denom = mask.sum().clamp_min(1.0)
        else:
            valid = (targets != ignore_index)
            denom = valid.sum().clamp_min(1).to(H.dtype)

        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            logits = (H[s:e].float() @ W.float().T)          # [c,V]
            tgt = targets[s:e]
            per = F.cross_entropy(logits, tgt, ignore_index=ignore_index,
                                  reduction='none')            # [c]
            if mask is not None:
                per = per * mask[s:e].float()
            loss_sum = loss_sum + per.sum()

        loss = loss_sum / denom
        ctx.save_for_backward(H, W, targets, mask)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.denom = denom
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        H, W, targets, mask = ctx.saved_tensors
        chunk, ignore_index, denom = ctx.chunk, ctx.ignore_index, ctx.denom
        N, V = H.shape[0], W.shape[0]
        g = (grad_out / denom)
        grad_H = torch.zeros_like(H)
        grad_W = torch.zeros_like(W)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            Hc = H[s:e].float()
            logits = Hc @ W.float().T                          # [c,V]
            p = torch.softmax(logits, dim=-1)                  # [c,V]
            tgt = targets[s:e]
            valid = (tgt != ignore_index)
            # d/dlogits CE = softmax - onehot(target)
            safe_tgt = torch.where(valid, tgt, torch.zeros_like(tgt))
            p.scatter_add_(1, safe_tgt.unsqueeze(1),
                           -torch.ones_like(safe_tgt, dtype=p.dtype).unsqueeze(1))
            if mask is not None:
                p = p * (g * mask[s:e].float()).unsqueeze(1)
            else:
                p = p * g
            p = p * valid.unsqueeze(1).float()                 # zero ignored rows
            grad_H[s:e] = (p @ W.float()).to(grad_H.dtype)
            grad_W += (p.T @ Hc).to(grad_W.dtype)
        return grad_H, grad_W, None, None, None, None


def fused_linear_cross_entropy(
    H: torch.Tensor,
    W: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    chunk: int = 4096,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean cross-entropy of (H @ W.T) vs targets, without materializing logits.

    H:[N,K] hidden (concat real/imag), W:[V,K] weights (concat real/imag),
    targets:[N], optional mask:[N] (1=count,0=skip). Returns scalar loss.
    """
    return _FusedLinearCE.apply(H, W, targets, mask, chunk, ignore_index)


@torch.no_grad()
def linear_ce_stats(H, W, targets, mask=None, chunk: int = 4096, ignore_index: int = -100):
    """No-grad eval stats without materializing [N,V] logits.

    Returns (loss_sum, correct, tokens) as python floats, where loss_sum and
    correct are summed over counted tokens (mask==1, or targets!=ignore_index).
    """
    N = H.shape[0]
    loss_sum = 0.0
    correct = 0.0
    tokens = 0.0
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        logits = (H[s:e].float() @ W.float().T)
        tgt = targets[s:e]
        per = F.cross_entropy(logits, tgt, ignore_index=ignore_index, reduction='none')
        pred = logits.argmax(dim=-1)
        corr = (pred == tgt).float()
        if mask is not None:
            m = mask[s:e].float()
        else:
            m = (tgt != ignore_index).float()
        loss_sum += (per * m).sum().item()
        correct += (corr * m).sum().item()
        tokens += m.sum().item()
    return loss_sum, correct, tokens


def _test():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    N, K, V = 200, 48, 512
    H = torch.randn(N, K, requires_grad=True)
    W = torch.randn(V, K, requires_grad=True)
    targets = torch.randint(0, V, (N,))
    targets[::7] = -100  # some ignored

    # reference
    Hr = H.detach().clone().requires_grad_(True)
    Wr = W.detach().clone().requires_grad_(True)
    logits = Hr @ Wr.T
    ref = F.cross_entropy(logits, targets, ignore_index=-100)
    ref.backward()

    loss = fused_linear_cross_entropy(H, W, targets, chunk=32)
    loss.backward()

    print(f"loss   diff = {(loss - ref).abs().item():.2e}")
    print(f"grad_H diff = {(H.grad - Hr.grad).abs().max().item():.2e}")
    print(f"grad_W diff = {(W.grad - Wr.grad).abs().max().item():.2e}")

    # masked variant vs manual masked reference
    Hm = H.detach().clone().requires_grad_(True)
    Wm = W.detach().clone().requires_grad_(True)
    mask = (torch.rand(N) > 0.3).double()
    tg = torch.randint(0, V, (N,))
    lg = Hm @ Wm.T
    per = F.cross_entropy(lg, tg, reduction='none')
    ref_m = (per * mask).sum() / mask.sum().clamp_min(1)
    ref_m.backward()
    Hm2 = H.detach().clone().requires_grad_(True)
    Wm2 = W.detach().clone().requires_grad_(True)
    lm = fused_linear_cross_entropy(Hm2, Wm2, tg, mask=mask, chunk=32, ignore_index=-1)
    lm.backward()
    print(f"[mask] loss diff = {(lm - ref_m).abs().item():.2e}")
    print(f"[mask] grad_H diff = {(Hm2.grad - Hm.grad).abs().max().item():.2e}")
    print(f"[mask] grad_W diff = {(Wm2.grad - Wm.grad).abs().max().item():.2e}")


if __name__ == '__main__':
    _test()
