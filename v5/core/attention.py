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
import warnings
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..init import InitStrategy

from .complex import ComplexLinear, ComplexNorm, cabs, to_real

try:
    import xformers.ops as xops
    from xformers.ops.fmha.attn_bias import LowerTriangularFromBottomRightMask
    HAS_XFORMERS = True
except Exception:
    xops = None
    LowerTriangularFromBottomRightMask = None
    HAS_XFORMERS = False


@dataclass
class AttentionKVCache:
    """Inference-only KV cache for one attention layer."""
    keys: torch.Tensor     # [B, H, L_cache, head_dim, 2]
    values: torch.Tensor   # [B, H, L_cache, head_dim, 2]


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
        attention_backend: str = 'native',
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        # Compute attention in exact query chunks to avoid materializing [L, L].
        self.max_chunk_size = 256
        self.scale = self.head_dim ** -0.5
        self.attention_backend = attention_backend
        self._xformers_fallback_reason: Optional[str] = None
        self._xformers_bias = None

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
        out, _ = self.forward_with_cache(z, cache=None, use_cache=False)
        return out

    def forward_with_cache(
        self,
        z: torch.Tensor,
        cache: Optional[AttentionKVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[AttentionKVCache]]:
        """
        Attention forward with optional inference KV cache.

        `use_cache=True` enables returning a cache for prompt prefill and
        incremental decoding. The no-cache path remains the reference
        implementation used during training.
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

        if use_cache:
            out, new_cache = self._apply_attention_with_cache(q, k, v, cache)
        else:
            out = self._apply_attention(q, k, v)
            new_cache = None

        # Reshape back
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(B, L, D, 2)

        return self.W_out(out), new_cache

    def _apply_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if self._should_use_xformers(q):
            try:
                return self._xformers_attention(q, k, v)
            except Exception as exc:
                self._disable_xformers(exc)
        return self._local_attention(q, k, v)

    def _apply_attention_with_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache: Optional[AttentionKVCache],
    ) -> tuple[torch.Tensor, AttentionKVCache]:
        keep = max(self.window_size - 1, 0)

        if cache is not None and q.shape[2] != 1:
            raise ValueError("Attention cache can only be updated incrementally with query length 1")

        if cache is None:
            out = self._apply_attention(q, k, v)
            new_cache = self._truncate_cache(k, v, keep)
            return out, new_cache

        if cache.keys.shape[:2] != k.shape[:2] or cache.values.shape[:2] != v.shape[:2]:
            raise ValueError("Attention cache batch/head shape mismatch")

        k_all = torch.cat([cache.keys, k], dim=2)
        v_all = torch.cat([cache.values, v], dim=2)

        if self._should_use_xformers(q):
            try:
                out = self._xformers_cached_attention(q, k_all, v_all)
            except Exception as exc:
                self._disable_xformers(exc)
                out = self._cached_attention(q, k_all, v_all)
        else:
            out = self._cached_attention(q, k_all, v_all)

        new_cache = self._truncate_cache(k_all, v_all, keep)
        return out, new_cache

    def _truncate_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        keep: int,
    ) -> AttentionKVCache:
        if keep <= 0:
            return AttentionKVCache(keys=k[:, :, :0], values=v[:, :, :0])
        return AttentionKVCache(keys=k[:, :, -keep:], values=v[:, :, -keep:])

    def _should_use_xformers(self, q: torch.Tensor) -> bool:
        if self.attention_backend == 'native':
            return False
        if self._xformers_fallback_reason is not None:
            return False
        if not HAS_XFORMERS:
            if self.attention_backend == 'xformers':
                self._disable_xformers(RuntimeError('xformers is not importable in this environment'))
            return False
        if q.device.type != 'cuda':
            if self.attention_backend == 'xformers':
                self._disable_xformers(RuntimeError('xformers attention requires CUDA tensors'))
            return False
        return True

    def _disable_xformers(self, exc: Exception) -> None:
        if self._xformers_fallback_reason is not None:
            return
        self._xformers_fallback_reason = str(exc)
        warnings.warn(
            "PhaseAttention xformers backend disabled; falling back to native exact attention. "
            f"Reason: {exc}"
        )

    def _get_xformers_bias(self):
        # Keep the native path below as the exact reference implementation.
        # If xformers behaves oddly on a driver/build, switching
        # `attention_backend` back to `native` reverts the code path cleanly.
        if self._xformers_bias is None:
            self._xformers_bias = (
                LowerTriangularFromBottomRightMask()
                .make_local_attention(self.window_size)
            )
        return self._xformers_bias

    def _xformers_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, HD, _ = q.shape

        q_cat = torch.cat([q[..., 0], q[..., 1]], dim=-1).permute(0, 2, 1, 3).contiguous()
        k_cat = torch.cat([k[..., 0], k[..., 1]], dim=-1).permute(0, 2, 1, 3).contiguous()
        v_cat = torch.cat([v[..., 0], v[..., 1]], dim=-1).permute(0, 2, 1, 3).contiguous()

        out = xops.memory_efficient_attention(
            q_cat,
            k_cat,
            v_cat,
            attn_bias=self._get_xformers_bias(),
            p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )  # [B, L, H, 2*HD]

        out = out.permute(0, 2, 1, 3).contiguous()
        out_real, out_imag = out.split(HD, dim=-1)
        return torch.stack([out_real, out_imag], dim=-1)

    def _xformers_cached_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        _, _, _, HD, _ = q.shape

        q_cat = torch.cat([q[..., 0], q[..., 1]], dim=-1).permute(0, 2, 1, 3).contiguous()
        k_cat = torch.cat([k[..., 0], k[..., 1]], dim=-1).permute(0, 2, 1, 3).contiguous()
        v_cat = torch.cat([v[..., 0], v[..., 1]], dim=-1).permute(0, 2, 1, 3).contiguous()

        # The cache is already trimmed to the valid local history window plus the
        # current token, so no additional causal/local mask is needed here.
        out = xops.memory_efficient_attention(
            q_cat,
            k_cat,
            v_cat,
            attn_bias=None,
            p=0.0,
            scale=self.scale,
        )
        out = out.permute(0, 2, 1, 3).contiguous()
        out_real, out_imag = out.split(HD, dim=-1)
        return torch.stack([out_real, out_imag], dim=-1)

    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exact causal sliding-window attention without materializing [L, L].

        We process query positions in chunks and only fetch the key/value span
        that can affect that chunk. The math is identical to the dense masked
        implementation; only the execution strategy changes.
        """
        _, _, L, _, _ = q.shape
        device = q.device
        chunk_size = min(L, self.window_size, self.max_chunk_size)

        q_real = q[..., 0]
        q_imag = q[..., 1]
        k_real = k[..., 0]
        k_imag = k[..., 1]
        v_real = v[..., 0]
        v_imag = v[..., 1]

        out_real_chunks = []
        out_imag_chunks = []

        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            key_start = max(0, start - self.window_size + 1)

            q_real_chunk = q_real[:, :, start:end, :]   # [B, H, C, HD]
            q_imag_chunk = q_imag[:, :, start:end, :]
            k_real_chunk = k_real[:, :, key_start:end, :]  # [B, H, K, HD]
            k_imag_chunk = k_imag[:, :, key_start:end, :]
            v_real_chunk = v_real[:, :, key_start:end, :]
            v_imag_chunk = v_imag[:, :, key_start:end, :]

            # Phase-aware score = Re(q * conj(k)) / sqrt(d)
            scores = (
                (q_real_chunk.unsqueeze(-2) * k_real_chunk.unsqueeze(-3)).sum(dim=-1) +
                (q_imag_chunk.unsqueeze(-2) * k_imag_chunk.unsqueeze(-3)).sum(dim=-1)
            ) * self.scale  # [B, H, C, K]

            mask = self._chunk_mask(start, end, key_start, device)
            scores = scores.masked_fill(~mask, float('-inf'))

            weights = F.softmax(scores, dim=-1)
            if self.training:
                weights = self.dropout(weights)

            out_real_chunks.append(torch.matmul(weights, v_real_chunk))
            out_imag_chunks.append(torch.matmul(weights, v_imag_chunk))

        out_real = torch.cat(out_real_chunks, dim=2)
        out_imag = torch.cat(out_imag_chunks, dim=2)
        return torch.stack([out_real, out_imag], dim=-1)

    def _cached_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Attention for incremental decoding with a pre-trimmed valid KV cache."""
        scores = (
            (q[..., 0].unsqueeze(-2) * k[..., 0].unsqueeze(-3)).sum(dim=-1) +
            (q[..., 1].unsqueeze(-2) * k[..., 1].unsqueeze(-3)).sum(dim=-1)
        ) * self.scale  # [B, H, 1, K]

        weights = F.softmax(scores, dim=-1)
        out_real = torch.matmul(weights, v[..., 0])
        out_imag = torch.matmul(weights, v[..., 1])
        return torch.stack([out_real, out_imag], dim=-1)

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
