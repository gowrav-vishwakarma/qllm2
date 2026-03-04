"""
PhaseAttention: complex-valued sliding window attention.

Score = Re(q * conj(k)) / sqrt(d) captures both magnitude similarity
AND phase alignment in a single dot product.

A single complex attention head does the work of two real heads:
one for magnitude similarity, one for directional alignment.

Used sparsely (every K-th layer) for content-addressable retrieval
that pure recurrence cannot provide.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..init import InitStrategy

from .complex import ComplexLinear, ComplexNorm, cabs, to_real


class PhaseAttention(nn.Module):
    """
    Complex-valued multi-head attention with sliding window.

    Queries, keys, values are all complex. The attention score uses
    Re(q * conj(k)) which naturally measures both magnitude and phase
    alignment. Values are complex and weighted-summed with real weights.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 256,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.W_q = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.W_k = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.W_v = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.W_out = ComplexLinear(dim, dim, bias=False, initializer=initializer)

        self.norm = ComplexNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, L, dim, 2] -> [B, L, dim, 2]
        """
        B, L, D, _ = z.shape
        H = self.num_heads
        HD = self.head_dim

        q = self.W_q(z)  # [B, L, D, 2]
        k = self.W_k(z)
        v = self.W_v(z)

        # Reshape to [B, H, L, HD, 2]
        q = q.view(B, L, H, HD, 2).permute(0, 2, 1, 3, 4)
        k = k.view(B, L, H, HD, 2).permute(0, 2, 1, 3, 4)
        v = v.view(B, L, H, HD, 2).permute(0, 2, 1, 3, 4)

        # Phase-aware attention score: Re(q * conj(k)) / sqrt(d)
        # = (q_r * k_r + q_i * k_i).sum(dim=-1) for the dot product over head_dim
        # q: [B, H, L, HD, 2], k: [B, H, L, HD, 2]
        scores = (
            (q[..., 0].unsqueeze(3) * k[..., 0].unsqueeze(2)).sum(dim=-1) +
            (q[..., 1].unsqueeze(3) * k[..., 1].unsqueeze(2)).sum(dim=-1)
        ) * self.scale  # [B, H, L_q, L_k]

        # Causal + sliding window mask
        mask = self._make_mask(L, self.window_size, scores.device)
        scores = scores + mask

        weights = F.softmax(scores, dim=-1)
        if self.training:
            weights = self.dropout(weights)

        # Apply weights to complex values
        # weights: [B, H, L, L], v: [B, H, L, HD, 2]
        out_real = torch.matmul(weights, v[..., 0])  # [B, H, L, HD]
        out_imag = torch.matmul(weights, v[..., 1])
        out = torch.stack([out_real, out_imag], dim=-1)  # [B, H, L, HD, 2]

        # Reshape back
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(B, L, D, 2)

        return self.W_out(out)

    @staticmethod
    def _make_mask(L: int, window: int, device: torch.device) -> torch.Tensor:
        """Causal + sliding window mask. Returns [1, 1, L, L] additive mask."""
        row = torch.arange(L, device=device).unsqueeze(1)
        col = torch.arange(L, device=device).unsqueeze(0)
        causal = col <= row
        windowed = (row - col) < window
        mask = causal & windowed
        return torch.where(mask, 0.0, float('-inf')).unsqueeze(0).unsqueeze(0)
