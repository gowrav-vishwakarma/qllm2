"""Flash-PAM: faster training-time evaluation of the PAM chunked dual form.

Background / constraint (see EXPERIMENTS_V11.md "Flash-PAM design constraint"):
custom-autograd Triton kernels historically fight `torch.compile` on this split-real
complex stack (they hit the `is_compiling()` guard or cause graph breaks), and removing
Triton in V7 gave +66% tok/s. So the *primary* Flash-PAM is a pure-PyTorch
**chunk-parallel** reformulation that inductor can fuse end-to-end — no custom autograd.

Baseline (`V11PAMLayer._forward_chunked_head`) loops over chunks in Python and runs the
expensive intra-chunk GEMMs (score [C,C] and AV [C,d]) *inside* that loop. Flash-PAM
instead:
  * folds the chunk axis into the batch and runs **all** intra-chunk GEMMs as one big
    batched matmul (better GPU utilisation, fewer launches), then
  * runs only the cheap d x d **state carry** sequentially over the n chunks.

It is numerically identical to `_forward_chunked_head` (same math, just regrouped), so
gradients match and it is a drop-in. Complex tensors are split-real `[..., 2]` throughout.

Public:
    flash_pam_chunked_head(q, k, v_prime, gamma, d, chunk_size) -> (y, S)
        q,k,v_prime: [B,H,T,d,2] (q NOT pre-scaled; scaled by d**-0.5 inside, matching
        the baseline). gamma: [B,H,T] head-scalar decay in (0,1].
        Returns y [B,H,T,d,2] and final state S [B,H,d,d,2].
"""

import math
import torch
from torch import Tensor


def _chunk_decay_matrices(gamma_bn: Tensor, C: int):
    """gamma_bn: [N, C] -> (D, cum_decay, total_decay, D_last).

    D[t,s]      = prod_{s<u<=t} gamma   (lower-tri, 0 above diagonal)   [N,C,C]
    cum_decay[t]= prod_{0<=u<=t} gamma  (inclusive)                     [N,C]
    total_decay = cum_decay[:, -1]                                      [N]
    D_last[s]   = prod_{s<u<=C-1} gamma = total/cum_excl                [N,C]
    """
    # decay is precision-sensitive: upcast only low precision (bf16/fp16) to fp32,
    # otherwise keep dtype (so fp64 correctness checks stay exact).
    cdtype = torch.float32 if gamma_bn.dtype in (torch.bfloat16, torch.float16) else gamma_bn.dtype
    log_g = torch.log(gamma_bn.to(cdtype) + 1e-6)
    Cc = torch.cumsum(-log_g, dim=-1)                       # [N,C], increasing
    log_D = (Cc.unsqueeze(-1) - Cc.unsqueeze(-2)).transpose(-1, -2)  # [t,s]=C[s]-C[t]
    causal = torch.tril(torch.ones(C, C, device=gamma_bn.device))
    log_D = log_D * causal + (1 - causal) * (-1e4)
    D = torch.exp(log_D.clamp(max=0.0))
    cum_decay = torch.exp(torch.cumsum(log_g, dim=-1))      # inclusive prod
    total_decay = cum_decay[:, -1]
    D_last = D[:, -1, :]                                    # [N,C]
    return D, cum_decay, total_decay, D_last


def flash_pam_chunked_head(
    q: Tensor, k: Tensor, v_prime: Tensor, gamma: Tensor, d: int, chunk_size: int,
):
    """Chunk-parallel equivalent of V11PAMLayer._forward_chunked_head (head decay)."""
    B, H, T = q.shape[:3]
    C = chunk_size
    scale = d ** -0.5
    n = (T + C - 1) // C
    Tpad = n * C
    pad = Tpad - T

    q_s = q * scale
    if pad:
        # pad keys/values with 0 (contribute nothing), gamma with 1 (log 0).
        q_s = torch.nn.functional.pad(q_s, (0, 0, 0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        v_prime = torch.nn.functional.pad(v_prime, (0, 0, 0, 0, 0, pad))
        gamma = torch.nn.functional.pad(gamma, (0, pad), value=1.0)

    # [B,H,n,C,...] -> fold (B,H,n) into one batch axis N.
    N = B * H * n
    qc = q_s.view(B, H, n, C, d, 2).reshape(N, C, d, 2)
    kc = k.view(B, H, n, C, d, 2).reshape(N, C, d, 2)
    vc = v_prime.view(B, H, n, C, d, 2).reshape(N, C, d, 2)
    gc = gamma.view(B, H, n, C).reshape(N, C)

    D, cum_decay, total_decay, D_last = _chunk_decay_matrices(gc, C)  # batched over N

    qr, qi = qc[..., 0], qc[..., 1]
    kr, ki = kc[..., 0], kc[..., 1]
    vr, vi = vc[..., 0], vc[..., 1]

    # intra-chunk complex conjugate score q . conj(k), masked+decayed by D
    wr = torch.bmm(qr, kr.transpose(-1, -2)) + torch.bmm(qi, ki.transpose(-1, -2))
    wi = torch.bmm(qi, kr.transpose(-1, -2)) - torch.bmm(qr, ki.transpose(-1, -2))
    Df = D.to(wr.dtype)
    ar, ai = wr * Df, wi * Df
    yr = torch.bmm(ar, vr) - torch.bmm(ai, vi)
    yi = torch.bmm(ar, vi) + torch.bmm(ai, vr)
    y_intra = torch.stack([yr, yi], dim=-1)                # [N,C,d,2]

    # per-chunk state block: S_block[i,j] = sum_s (v_s * D_last_s)[i] conj(k_s)[j]
    dl = D_last.unsqueeze(-1).to(vr.dtype)
    wv_r, wv_i = vr * dl, vi * dl
    sr = torch.bmm(wv_r.transpose(-1, -2), kr) + torch.bmm(wv_i.transpose(-1, -2), ki)
    si = torch.bmm(wv_i.transpose(-1, -2), kr) - torch.bmm(wv_r.transpose(-1, -2), ki)
    S_block = torch.stack([sr, si], dim=-1)                # [N,d,d,2]

    # reshape back to [B,H,n,...] for the sequential state carry
    y_intra = y_intra.view(B, H, n, C, d, 2)
    S_block = S_block.view(B, H, n, d, d, 2)
    cum = cum_decay.view(B, H, n, C).to(q.dtype)
    total = total_decay.view(B, H, n).to(q.dtype)
    qcs = q_s.view(B, H, n, C, d, 2)

    outputs = []
    S = q.new_zeros(B, H, d, d, 2)
    for c in range(n):
        y_c = y_intra[:, :, c]
        if c > 0:
            Sr, Si = S[..., 0], S[..., 1]
            qr_c, qi_c = qcs[:, :, c, ..., 0], qcs[:, :, c, ..., 1]   # [B,H,C,d]
            Sq_r = (Sr @ qr_c.transpose(-1, -2) - Si @ qi_c.transpose(-1, -2)).transpose(-1, -2)
            Sq_i = (Sr @ qi_c.transpose(-1, -2) + Si @ qr_c.transpose(-1, -2)).transpose(-1, -2)
            cd = cum[:, :, c].unsqueeze(-1)
            y_c = y_c + torch.stack([Sq_r * cd, Sq_i * cd], dim=-1)
        outputs.append(y_c)
        S = S * total[:, :, c][..., None, None, None] + S_block[:, :, c]

    y = torch.cat(outputs, dim=2)                          # [B,H,Tpad,C? ] -> [B,H,Tpad,d,2]
    if pad:
        y = y[:, :, :T]
    return y, S
