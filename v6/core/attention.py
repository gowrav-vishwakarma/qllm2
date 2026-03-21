"""
PhaseAttention: complex-valued sliding window attention.

Two modes:
  'softmax'      — Standard: Re(q * conj(k)) / sqrt(d) as score, softmax weights,
                   independent real/imag value aggregation. (v5 compatible)
  'interference' — Phase Interference Attention (PIA): full complex score
                   Q @ K* / sqrt(d), magnitude-normalized complex weights,
                   full complex value aggregation. No softmax.

Supports complex RoPE on Q,K and fused QKV projections (matching PAM).
Used sparsely (every K-th layer or last layer only) for content-addressable
retrieval that pure recurrence cannot provide.
Disabled by default -- the model is attention-free unless explicitly enabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..init import InitStrategy

from .complex import ComplexLinear, ComplexNorm, cmul, cabs, to_real, build_rope_cache


class PhaseAttention(nn.Module):
    """
    Complex-valued multi-head attention with sliding window.

    mode='softmax':
        Score = Re(q * conj(k)), softmax weights, separate real/imag aggregation.
    mode='interference':
        Score = q * conj(k) (full complex), magnitude-normalized complex weights,
        full complex matmul for value aggregation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 256,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
        mode: str = 'softmax',
        rope: bool = False,
        fused_qkv: bool = False,
        rope_max_len: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.max_chunk_size = 256
        self.scale = self.head_dim ** -0.5
        self.mode = mode
        self.rope_enabled = rope
        self.fused_qkv = fused_qkv

        if fused_qkv:
            self.W_qkv = ComplexLinear(dim, 3 * dim, bias=False, initializer=initializer)
        else:
            self.W_q = ComplexLinear(dim, dim, bias=False, initializer=initializer)
            self.W_k = ComplexLinear(dim, dim, bias=False, initializer=initializer)
            self.W_v = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.W_out = ComplexLinear(dim, dim, bias=False, initializer=initializer)

        self.norm = ComplexNorm(dim)
        self.dropout = nn.Dropout(dropout)

        if rope:
            self.register_buffer(
                'rope_cache', build_rope_cache(rope_max_len, self.head_dim),
                persistent=False,
            )

    def _project_qkv(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input to Q, K, V and reshape to [B, H, L, HD, 2]."""
        B, L, D, _ = z.shape
        H = self.num_heads
        HD = self.head_dim

        if self.fused_qkv:
            qkv = self.W_qkv(z).view(B, L, 3, H, HD, 2)
            q = qkv[:, :, 0].permute(0, 2, 1, 3, 4).contiguous()
            k = qkv[:, :, 1].permute(0, 2, 1, 3, 4).contiguous()
            v = qkv[:, :, 2].permute(0, 2, 1, 3, 4).contiguous()
        else:
            q = self.W_q(z).view(B, L, H, HD, 2).permute(0, 2, 1, 3, 4)
            k = self.W_k(z).view(B, L, H, HD, 2).permute(0, 2, 1, 3, 4)
            v = self.W_v(z).view(B, L, H, HD, 2).permute(0, 2, 1, 3, 4)

        return q, k, v

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, L: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply complex RoPE to Q and K. q,k: [B, H, L, HD, 2]."""
        if L > self.rope_cache.shape[0]:
            self.register_buffer(
                'rope_cache',
                build_rope_cache(L * 2, self.head_dim).to(q.device),
                persistent=False,
            )
        pos = self.rope_cache[:L].to(dtype=q.dtype)  # [L, HD, 2]
        q = cmul(q, pos)
        k = cmul(k, pos)
        return q, k

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, L, dim, 2] -> [B, L, dim, 2]"""
        B, L, D, _ = z.shape

        q, k, v = self._project_qkv(z)

        if self.rope_enabled:
            q, k = self._apply_rope(q, k, L)

        if self.mode == 'interference':
            out = self._interference_attention(q, k, v)
        else:
            out = self._softmax_attention(q, k, v)

        out = out.permute(0, 2, 1, 3, 4).contiguous().view(B, L, D, 2)
        return self.W_out(out)

    # ------------------------------------------------------------------
    # Softmax mode (backward-compatible v5 path)
    # ------------------------------------------------------------------

    def _softmax_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked causal sliding-window attention with softmax weights."""
        _, _, L, _, _ = q.shape
        device = q.device
        chunk_size = min(L, self.window_size, self.max_chunk_size)

        qr, qi = q[..., 0], q[..., 1]
        kr, ki = k[..., 0], k[..., 1]
        vr, vi = v[..., 0], v[..., 1]

        out_r_chunks = []
        out_i_chunks = []

        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            key_start = max(0, start - self.window_size + 1)

            qr_c = qr[:, :, start:end, :]
            qi_c = qi[:, :, start:end, :]
            kr_c = kr[:, :, key_start:end, :]
            ki_c = ki[:, :, key_start:end, :]
            vr_c = vr[:, :, key_start:end, :]
            vi_c = vi[:, :, key_start:end, :]

            # Re(q * conj(k)) / sqrt(d)
            scores = (
                (qr_c.unsqueeze(-2) * kr_c.unsqueeze(-3)).sum(dim=-1) +
                (qi_c.unsqueeze(-2) * ki_c.unsqueeze(-3)).sum(dim=-1)
            ) * self.scale

            mask = self._chunk_mask(start, end, key_start, device)
            scores = scores.masked_fill(~mask, float('-inf'))

            weights = F.softmax(scores, dim=-1)
            if self.training:
                weights = self.dropout(weights)

            out_r_chunks.append(torch.matmul(weights, vr_c))
            out_i_chunks.append(torch.matmul(weights, vi_c))

        out_r = torch.cat(out_r_chunks, dim=2)
        out_i = torch.cat(out_i_chunks, dim=2)
        return torch.stack([out_r, out_i], dim=-1)

    # ------------------------------------------------------------------
    # Interference mode (Phase Interference Attention)
    # ------------------------------------------------------------------

    def _interference_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chunked causal sliding-window attention with complex phase interference.

        Full complex score W = (Q @ K*) / sqrt(d), magnitude-normalized,
        then full complex matmul with values.
        """
        _, _, L, _, _ = q.shape
        device = q.device
        chunk_size = min(L, self.window_size, self.max_chunk_size)

        qr, qi = q[..., 0], q[..., 1]
        kr, ki = k[..., 0], k[..., 1]
        vr, vi = v[..., 0], v[..., 1]

        out_r_chunks = []
        out_i_chunks = []

        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            key_start = max(0, start - self.window_size + 1)

            qr_c = qr[:, :, start:end, :] * self.scale
            qi_c = qi[:, :, start:end, :] * self.scale
            kr_c = kr[:, :, key_start:end, :]
            ki_c = ki[:, :, key_start:end, :]
            vr_c = vr[:, :, key_start:end, :]
            vi_c = vi[:, :, key_start:end, :]

            # Full complex score: W = Q @ K* = (Qr + i·Qi)(Kr - i·Ki)
            #   Wr = Qr·Kr^T + Qi·Ki^T    (= Re(q · conj(k)))
            #   Wi = Qi·Kr^T - Qr·Ki^T    (= Im(q · conj(k)))
            wr = (qr_c.unsqueeze(-2) * kr_c.unsqueeze(-3)).sum(-1) + \
                 (qi_c.unsqueeze(-2) * ki_c.unsqueeze(-3)).sum(-1)
            wi = (qi_c.unsqueeze(-2) * kr_c.unsqueeze(-3)).sum(-1) - \
                 (qr_c.unsqueeze(-2) * ki_c.unsqueeze(-3)).sum(-1)

            mask = self._chunk_mask(start, end, key_start, device)
            wr = wr.masked_fill(~mask, 0.0)
            wi = wi.masked_fill(~mask, 0.0)

            # Magnitude-normalize: A = W / (sum_j |W_j| + eps)
            # Magnitudes sum to 1 per query position; phases preserved.
            mag = torch.sqrt(wr * wr + wi * wi + 1e-8)
            mag_sum = mag.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            ar = wr / mag_sum
            ai = wi / mag_sum

            if self.training:
                drop_mask = self.dropout(
                    torch.ones(ar.shape[:-1], device=device)
                ).unsqueeze(-1)
                ar = ar * drop_mask
                ai = ai * drop_mask

            # Full complex matmul: Y = A @ V
            out_r_chunks.append(torch.matmul(ar, vr_c) - torch.matmul(ai, vi_c))
            out_i_chunks.append(torch.matmul(ar, vi_c) + torch.matmul(ai, vr_c))

        out_r = torch.cat(out_r_chunks, dim=2)
        out_i = torch.cat(out_i_chunks, dim=2)
        return torch.stack([out_r, out_i], dim=-1)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _chunk_mask(
        self,
        start: int,
        end: int,
        key_start: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Boolean mask for one query chunk, shaped [1, 1, C, K]."""
        query_pos = torch.arange(start, end, device=device).unsqueeze(1)
        key_pos = torch.arange(key_start, end, device=device).unsqueeze(0)
        valid = (key_pos <= query_pos) & (key_pos >= (query_pos - self.window_size + 1))
        return valid.unsqueeze(0).unsqueeze(0)
