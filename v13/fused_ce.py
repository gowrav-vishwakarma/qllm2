"""Fused (chunked) linear + cross-entropy for the tied complex LM head.

The V11 head is algebraically a single real matmul:

    logits = lm_real @ E_real.T + lm_imag @ E_imag.T
           = concat(lm_real, lm_imag) @ concat(E_real, E_imag).T
           = hidden_rows @ weight_matrix.T    hidden_rows:[N,2d]  weight_matrix:[V,2d]

For vocab V~50k and N=B*T~37k the logits tensor `[N, V]` (~4 GB fp32, plus an
equal-size softmax and grad) dominates training memory. This module computes the
mean/masked cross-entropy WITHOUT ever materializing the full `[N, V]` logits:

  * forward  processes hidden_rows in row-chunks, keeping only `[chunk, V]` live;
  * backward recomputes the per-chunk logits and accumulates gradients chunk by chunk.

Peak head memory drops from O(N*V) to O(chunk*V). Math is exact (fp32 reduction),
verified against F.cross_entropy in `_test`.
"""

from typing import Optional

import torch
import torch.nn.functional as F


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_rows, weight_matrix, targets, mask, chunk, ignore_index, return_nll):
        num_rows = hidden_rows.shape[0]
        loss_sum = hidden_rows.new_zeros(())
        nll_out = torch.zeros(num_rows, dtype=torch.float32, device=hidden_rows.device) if return_nll else None
        if mask is not None:
            denom = mask.sum().clamp_min(1.0)
        else:
            valid = (targets != ignore_index)
            denom = valid.sum().clamp_min(1).to(hidden_rows.dtype)

        for chunk_start in range(0, num_rows, chunk):
            chunk_end = min(chunk_start + chunk, num_rows)
            # Force fp32 for the head GEMM + CE: under autocast the matmul is
            # downcast to bf16 (8-bit mantissa) even with .float() inputs, which
            # quantizes the NLL to ~0.03. The loss sum and the NLL byproduct
            # must stay exact fp32 (fp32 GEMM here is ~4% of step time).
            with torch.amp.autocast(device_type=hidden_rows.device.type, enabled=False):
                logits = (hidden_rows[chunk_start:chunk_end].float() @ weight_matrix.float().T)
                target_chunk = targets[chunk_start:chunk_end]
                per_token_loss = F.cross_entropy(
                    logits, target_chunk, ignore_index=ignore_index, reduction='none',
                )
            if nll_out is not None:
                # Materialized byproduct: raw per-token CE (ignore rows 0.0),
                # captured BEFORE the mask multiply. fp32 no-grad leaf; O(1)
                # in vocab — consumers (gate-surprisal aux) reuse it instead
                # of re-running a second O(V) head GEMM.
                with torch.no_grad():
                    nll_out[chunk_start:chunk_end] = per_token_loss
            if mask is not None:
                per_token_loss = per_token_loss * mask[chunk_start:chunk_end].float()
            loss_sum = loss_sum + per_token_loss.sum()

        loss = loss_sum / denom
        ctx.save_for_backward(hidden_rows, weight_matrix, targets, mask)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.denom = denom
        if nll_out is not None:
            loss._nll = nll_out  # [N] fp32, detached
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden_rows, weight_matrix, targets, mask = ctx.saved_tensors
        chunk, ignore_index, denom = ctx.chunk, ctx.ignore_index, ctx.denom
        num_rows, vocab_size = hidden_rows.shape[0], weight_matrix.shape[0]
        grad_scale = (grad_output / denom)
        grad_hidden = torch.zeros_like(hidden_rows)
        grad_weight = torch.zeros_like(weight_matrix)
        for chunk_start in range(0, num_rows, chunk):
            chunk_end = min(chunk_start + chunk, num_rows)
            hidden_chunk = hidden_rows[chunk_start:chunk_end].float()
            logits = hidden_chunk @ weight_matrix.float().T
            softmax_probs = torch.softmax(logits, dim=-1)
            target_chunk = targets[chunk_start:chunk_end]
            valid = (target_chunk != ignore_index)
            safe_targets = torch.where(valid, target_chunk, torch.zeros_like(target_chunk))
            softmax_probs.scatter_add_(
                1, safe_targets.unsqueeze(1),
                -torch.ones_like(safe_targets, dtype=softmax_probs.dtype).unsqueeze(1),
            )
            if mask is not None:
                softmax_probs = softmax_probs * (grad_scale * mask[chunk_start:chunk_end].float()).unsqueeze(1)
            else:
                softmax_probs = softmax_probs * grad_scale
            softmax_probs = softmax_probs * valid.unsqueeze(1).float()
            grad_hidden[chunk_start:chunk_end] = (softmax_probs @ weight_matrix.float()).to(grad_hidden.dtype)
        return grad_hidden, grad_weight, None, None, None, None, None


def fused_linear_cross_entropy(
    hidden_rows: torch.Tensor,
    weight_matrix: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    chunk: int = 4096,
    ignore_index: int = -100,
    return_nll: bool = False,
) -> torch.Tensor:
    """Mean cross-entropy of (hidden_rows @ weight_matrix.T) vs targets.

    return_nll=True attaches the exact per-token NLL this forward already
    computes (fp32, no-grad, [N], ignore rows 0.0) to the returned loss as
    ``loss._nll`` — a materialized-intermediate byproduct with no extra
    head pass (O(1) in vocab).
    """
    return _FusedLinearCE.apply(
        hidden_rows, weight_matrix, targets, mask, chunk, ignore_index, return_nll,
    )


@torch.no_grad()
def linear_ce_per_token(
    hidden_rows, weight_matrix, targets, chunk: int = 4096, ignore_index: int = -100,
):
    """Detached per-row cross-entropy `[N]` (surprisal), no [N,V] materialization.

    Used as a stop-grad target for the gate-surprisal aux loss. ignore_index rows
    return 0.0 (they are masked out by the caller).
    """
    num_rows = hidden_rows.shape[0]
    out = hidden_rows.new_zeros(num_rows)
    for chunk_start in range(0, num_rows, chunk):
        chunk_end = min(chunk_start + chunk, num_rows)
        logits = (hidden_rows[chunk_start:chunk_end].float() @ weight_matrix.float().T)
        out[chunk_start:chunk_end] = F.cross_entropy(
            logits, targets[chunk_start:chunk_end],
            ignore_index=ignore_index, reduction='none',
        )
    return out


@torch.no_grad()
def linear_ce_stats(
    hidden_rows, weight_matrix, targets, mask=None, chunk: int = 4096, ignore_index: int = -100,
):
    """No-grad eval stats without materializing [N,V] logits."""
    num_rows = hidden_rows.shape[0]
    loss_sum = 0.0
    correct = 0.0
    tokens = 0.0
    for chunk_start in range(0, num_rows, chunk):
        chunk_end = min(chunk_start + chunk, num_rows)
        logits = (hidden_rows[chunk_start:chunk_end].float() @ weight_matrix.float().T)
        target_chunk = targets[chunk_start:chunk_end]
        per_token_loss = F.cross_entropy(logits, target_chunk, ignore_index=ignore_index, reduction='none')
        predictions = logits.argmax(dim=-1)
        correct_mask = (predictions == target_chunk).float()
        if mask is not None:
            token_mask = mask[chunk_start:chunk_end].float()
        else:
            token_mask = (target_chunk != ignore_index).float()
        loss_sum += (per_token_loss * token_mask).sum().item()
        correct += (correct_mask * token_mask).sum().item()
        tokens += token_mask.sum().item()
    return loss_sum, correct, tokens


def _test():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    num_rows, feature_dim, vocab_size = 200, 48, 512
    hidden_rows = torch.randn(num_rows, feature_dim, requires_grad=True)
    weight_matrix = torch.randn(vocab_size, feature_dim, requires_grad=True)
    targets = torch.randint(0, vocab_size, (num_rows,))
    targets[::7] = -100

    hidden_ref = hidden_rows.detach().clone().requires_grad_(True)
    weight_ref = weight_matrix.detach().clone().requires_grad_(True)
    logits = hidden_ref @ weight_ref.T
    ref_loss = F.cross_entropy(logits, targets, ignore_index=-100)
    ref_loss.backward()

    loss = fused_linear_cross_entropy(hidden_rows, weight_matrix, targets, chunk=32)
    loss.backward()

    print(f"loss   diff = {(loss - ref_loss).abs().item():.2e}")
    print(f"grad_H diff = {(hidden_rows.grad - hidden_ref.grad).abs().max().item():.2e}")
    print(f"grad_W diff = {(weight_matrix.grad - weight_ref.grad).abs().max().item():.2e}")

    hidden_masked = hidden_rows.detach().clone().requires_grad_(True)
    weight_masked = weight_matrix.detach().clone().requires_grad_(True)
    token_mask = (torch.rand(num_rows) > 0.3).double()
    targets_masked = torch.randint(0, vocab_size, (num_rows,))
    logits_masked = hidden_masked @ weight_masked.T
    per_token = F.cross_entropy(logits_masked, targets_masked, reduction='none')
    ref_masked = (per_token * token_mask).sum() / token_mask.sum().clamp_min(1)
    ref_masked.backward()
    hidden_masked2 = hidden_rows.detach().clone().requires_grad_(True)
    weight_masked2 = weight_matrix.detach().clone().requires_grad_(True)
    loss_masked = fused_linear_cross_entropy(
        hidden_masked2, weight_masked2, targets_masked, mask=token_mask, chunk=32, ignore_index=-1,
    )
    loss_masked.backward()
    print(f"[mask] loss diff = {(loss_masked - ref_masked).abs().item():.2e}")
    print(f"[mask] grad_H diff = {(hidden_masked2.grad - hidden_masked.grad).abs().max().item():.2e}")
    print(f"[mask] grad_W diff = {(weight_masked2.grad - weight_masked.grad).abs().max().item():.2e}")


if __name__ == '__main__':
    _test()
