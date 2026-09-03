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

Peak head memory drops from O(N*V) to O(chunk*V). The softmax / loss / NLL
reduction is always fp32; the four head GEMMs (logits forward, logits
recompute, grad_hidden, grad_weight) run in ``gemm_dtype`` — bf16 under
autocast (the standard bf16-LLM head: logits carry ~1e-3 relative error, the
loss value ~1e-5), or fp32 for the exact path (``gemm_dtype=torch.float32``,
used for validation and by ``_test`` against F.cross_entropy). On the
real-101M step the fp32 head was 33% of GPU time.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

_TRITON_CE = os.environ.get("V13S_KERNEL", "1") == "1"


def _resolve_gemm_dtype(hidden_rows, gemm_dtype):
    if gemm_dtype is not None:
        return gemm_dtype
    dev = hidden_rows.device.type
    if torch.is_autocast_enabled(dev):
        return torch.get_autocast_dtype(dev)
    return hidden_rows.dtype


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_rows, weight_matrix, targets, mask, chunk, ignore_index,
                return_nll, gemm_dtype):
        num_rows = hidden_rows.shape[0]
        loss_sum = hidden_rows.new_zeros((), dtype=torch.float32)
        nll_out = torch.zeros(num_rows, dtype=torch.float32, device=hidden_rows.device) if return_nll else None
        if mask is not None:
            denom = mask.sum().clamp_min(1.0).float()
        else:
            valid = (targets != ignore_index)
            denom = valid.sum().clamp_min(1).float()
        weight_g = weight_matrix.to(gemm_dtype)

        for chunk_start in range(0, num_rows, chunk):
            chunk_end = min(chunk_start + chunk, num_rows)
            # The GEMM dtype is decided here, not by autocast; the CE itself
            # is fp32 so the loss sum and the NLL byproduct are not quantized
            # beyond what the logits carry.
            with torch.amp.autocast(device_type=hidden_rows.device.type, enabled=False):
                logits = (hidden_rows[chunk_start:chunk_end].to(gemm_dtype) @ weight_g.T).float()
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

        loss = loss_sum / denom                     # fp32 scalar, whatever the GEMM dtype
        ctx.save_for_backward(hidden_rows, weight_matrix, targets, mask)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.denom = denom
        ctx.gemm_dtype = gemm_dtype
        if nll_out is not None:
            loss._nll = nll_out  # [N] fp32, detached
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden_rows, weight_matrix, targets, mask = ctx.saved_tensors
        chunk, ignore_index, denom = ctx.chunk, ctx.ignore_index, ctx.denom
        gemm_dtype = ctx.gemm_dtype
        num_rows = hidden_rows.shape[0]
        grad_scale = (grad_output.float() / denom)
        grad_hidden = torch.empty_like(hidden_rows)
        grad_weight = torch.zeros(weight_matrix.shape, dtype=torch.float32,
                                  device=weight_matrix.device)
        weight_g = weight_matrix.to(gemm_dtype)
        for chunk_start in range(0, num_rows, chunk):
            chunk_end = min(chunk_start + chunk, num_rows)
            hidden_chunk = hidden_rows[chunk_start:chunk_end].to(gemm_dtype)
            logits = (hidden_chunk @ weight_g.T).float()
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
            softmax_probs = (softmax_probs * valid.unsqueeze(1).float()).to(gemm_dtype)
            grad_hidden[chunk_start:chunk_end] = (softmax_probs @ weight_g).to(grad_hidden.dtype)
            grad_weight += (softmax_probs.T @ hidden_chunk).float()
        return (grad_hidden, grad_weight.to(weight_matrix.dtype),
                None, None, None, None, None, None)


if HAS_TRITON:

    @triton.jit
    def _ce_rows_kernel(logits_ptr, stride_row, targets_ptr, nll_ptr, mask_ptr,
                        inv_denom_ptr, V, ignore_index,
                        HAS_MASK: tl.constexpr, NEED_GRAD: tl.constexpr,
                        BV: tl.constexpr):
        """One program per row: nll[row] = logsumexp(x) - x[target] (0 for
        ignore rows); with NEED_GRAD the row is overwritten IN PLACE by
        d(mean loss)/d(x) = (mask[row] / denom) * (softmax(x) - onehot)."""
        row = tl.program_id(0)
        base = logits_ptr + row.to(tl.int64) * stride_row
        target = tl.load(targets_ptr + row)
        valid = target != ignore_index
        t_safe = tl.where(valid, target, 0)

        m = float("-inf")
        s = 0.0
        for start in range(0, V, BV):
            offs = start + tl.arange(0, BV)
            x = tl.load(base + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
            m_new = tl.maximum(m, tl.max(x, axis=0))
            s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
            m = m_new
        lse = m + tl.log(s)
        x_t = tl.load(base + t_safe).to(tl.float32)
        tl.store(nll_ptr + row, tl.where(valid, lse - x_t, 0.0))

        if NEED_GRAD:
            w = tl.load(inv_denom_ptr)
            if HAS_MASK:
                w = w * tl.load(mask_ptr + row).to(tl.float32)
            w = tl.where(valid, w, 0.0)
            for start in range(0, V, BV):
                offs = start + tl.arange(0, BV)
                x = tl.load(base + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
                p = tl.exp(x - lse) * w
                p = tl.where(offs == t_safe, p - w, p)
                tl.store(base + offs, p.to(logits_ptr.dtype.element_ty), mask=offs < V)


def _ce_rows(logits, targets, nll, mask, inv_denom, ignore_index, need_grad):
    rows, V = logits.shape
    BV = min(8192, triton.next_power_of_2(V))
    _ce_rows_kernel[(rows,)](
        logits, logits.stride(0), targets, nll, mask if mask is not None else nll,
        inv_denom, V, ignore_index,
        HAS_MASK=mask is not None, NEED_GRAD=need_grad, BV=BV, num_warps=8,
    )


class _FusedLinearCETriton(torch.autograd.Function):
    """Forward computes the gradients too (Liger-style).

    Per chunk: logits = h @ W^T (gemm_dtype); one Triton pass turns the
    logits into per-row NLL and, in place, into d loss / d logits; then
    grad_h = dlogits @ W and grad_W += dlogits^T @ h while the chunk is
    hot. Three GEMMs and one elementwise pass instead of four GEMMs and
    ~10 fp32 [chunk, V] passes; backward is two scalar multiplies.
    """

    @staticmethod
    def forward(ctx, hidden_rows, weight_matrix, targets, mask, chunk, ignore_index,
                return_nll, gemm_dtype, need_grad):
        # dtypes are explicit below; autocast must not rewrite the GEMMs
        with torch.amp.autocast(device_type=hidden_rows.device.type, enabled=False):
            return _FusedLinearCETriton._forward(
                ctx, hidden_rows, weight_matrix, targets, mask, chunk, ignore_index,
                return_nll, gemm_dtype, need_grad)

    @staticmethod
    def _forward(ctx, hidden_rows, weight_matrix, targets, mask, chunk, ignore_index,
                 return_nll, gemm_dtype, need_grad):
        num_rows, feat = hidden_rows.shape
        vocab = weight_matrix.shape[0]
        if mask is not None:
            mask = mask.contiguous().float()
            denom = mask.sum().clamp_min(1.0)
        else:
            denom = (targets != ignore_index).sum().clamp_min(1).float()
        targets = targets.contiguous()
        h_g = hidden_rows.to(gemm_dtype)
        w_g = weight_matrix.to(gemm_dtype)
        nll = torch.empty(num_rows, dtype=torch.float32, device=hidden_rows.device)
        grad_h = grad_w = None
        if need_grad:
            grad_h = torch.empty_like(hidden_rows)
            grad_w = torch.zeros(vocab, feat, dtype=torch.float32, device=weight_matrix.device)
        # the 1/denom factor goes in here (device scalar, no sync);
        # grad_output is applied in backward
        inv_denom = (1.0 / denom).reshape(1)

        for s in range(0, num_rows, chunk):
            e = min(s + chunk, num_rows)
            logits = torch.mm(h_g[s:e], w_g.T)                      # [c, V] gemm_dtype
            _ce_rows(logits, targets[s:e], nll[s:e],
                     mask[s:e] if mask is not None else None,
                     inv_denom, ignore_index, need_grad)
            if need_grad:
                if grad_h.dtype == gemm_dtype:
                    torch.mm(logits, w_g, out=grad_h[s:e])
                else:
                    grad_h[s:e] = torch.mm(logits, w_g, out_dtype=grad_h.dtype)
                grad_w = torch.addmm(grad_w, logits.T, h_g[s:e], out_dtype=torch.float32)

        loss = ((nll * mask).sum() if mask is not None else nll.sum()) / denom
        ctx.grad_h, ctx.grad_w = grad_h, grad_w
        ctx.weight_dtype = weight_matrix.dtype
        if return_nll:
            loss._nll = nll  # [N] fp32, detached, raw (pre-mask) per-token CE
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_h, grad_w = ctx.grad_h, ctx.grad_w
        ctx.grad_h = ctx.grad_w = None
        g = grad_output.to(torch.float32)
        gh = (grad_h * g).to(grad_h.dtype) if grad_h is not None else None
        gw = (grad_w * g).to(ctx.weight_dtype) if grad_w is not None else None
        return gh, gw, None, None, None, None, None, None, None


def _use_triton_ce(hidden_rows) -> bool:
    return (HAS_TRITON and _TRITON_CE and hidden_rows.is_cuda
            and not torch.compiler.is_compiling())


def fused_linear_cross_entropy(
    hidden_rows: torch.Tensor,
    weight_matrix: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    chunk: int = 4096,
    ignore_index: int = -100,
    return_nll: bool = False,
    gemm_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Mean cross-entropy of (hidden_rows @ weight_matrix.T) vs targets.

    ``gemm_dtype`` is the dtype of the head GEMMs: None picks the autocast
    dtype when autocast is active (bf16 in training), else the hidden dtype;
    pass ``torch.float32`` for the exact path (validation, tests).

    On CUDA with Triton the Liger-style Function (gradients computed in the
    forward, one fused row pass) is used; elsewhere the chunked torch one.
    Both give the same loss, NLL byproduct and gradients.

    return_nll=True attaches the per-token NLL this forward already computes
    (fp32, no-grad, [N], ignore rows 0.0) to the returned loss as
    ``loss._nll`` — a materialized-intermediate byproduct with no extra
    head pass (O(1) in vocab).
    """
    gemm_dtype = _resolve_gemm_dtype(hidden_rows, gemm_dtype)
    if _use_triton_ce(hidden_rows):
        need_grad = torch.is_grad_enabled() and (
            hidden_rows.requires_grad or weight_matrix.requires_grad)
        return _FusedLinearCETriton.apply(
            hidden_rows, weight_matrix, targets, mask, chunk, ignore_index, return_nll,
            gemm_dtype, need_grad,
        )
    return _FusedLinearCE.apply(
        hidden_rows, weight_matrix, targets, mask, chunk, ignore_index, return_nll,
        gemm_dtype,
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
