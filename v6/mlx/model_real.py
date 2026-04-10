"""
RPAM (Real-valued Phase-Associative Memory) — MLX implementation.

Real-valued ablation of QPAM: architecturally identical but with all complex
operations replaced by standard real-valued equivalents.  The goal is to test
whether complex-valued computation provides a genuine advantage.

Dimension is set to 576 by default (with 9 heads) so that the total parameter
count roughly matches the complex QPAM at dim=384 (~100-120M parameters).
"""

import math
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────
# RoPE (standard real-valued)
# ──────────────────────────────────────────────────────────────

def build_rope_freqs(dim: int, max_len: int = 8192, base: float = 10000.0):
    """Precompute RoPE cos/sin tables."""
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(max_len, dtype=mx.float32)
    angles = mx.outer(t, freqs)  # [T, dim/2]
    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)
    cos_a = mx.repeat(cos_a, 2, axis=-1)[:, :dim]
    sin_a = mx.repeat(sin_a, 2, axis=-1)[:, :dim]
    return cos_a, sin_a


def apply_rope(x, cos_freqs, sin_freqs, offset: int = 0):
    """Standard RoPE: rotate pairs of dimensions.
    x: [..., T, d]
    """
    T = x.shape[-2]
    cos_f = cos_freqs[offset:offset + T]  # [T, d]
    sin_f = sin_freqs[offset:offset + T]
    # Interleaved rotation: for pair (x0,x1), rotate by theta
    # With repeat-style freqs, even/odd dims form pairs
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos_f1 = cos_f[..., 0::2]
    sin_f1 = sin_f[..., 0::2]
    y1 = x1 * cos_f1 - x2 * sin_f1
    y2 = x1 * sin_f1 + x2 * cos_f1
    # Interleave back
    # Stack along last dim then reshape
    shape = list(y1.shape)
    shape[-1] *= 2
    out = mx.zeros(shape)
    out = mx.concatenate([mx.expand_dims(y1, -1), mx.expand_dims(y2, -1)], axis=-1)
    out = mx.reshape(out, shape)
    return out


# ──────────────────────────────────────────────────────────────
# SwiGLU block (replaces ComplexGatedUnit)
# ──────────────────────────────────────────────────────────────

class SwiGLUBlock(nn.Module):
    """SwiGLU feed-forward block: standard real-valued replacement for CGU."""
    def __init__(self, dim: int, expand: int = 3):
        super().__init__()
        hidden = dim * expand
        self.up = nn.Linear(dim, hidden, bias=False)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


# ──────────────────────────────────────────────────────────────
# Real-valued PAM Layer
# ──────────────────────────────────────────────────────────────

class PAMLayerReal(nn.Module):
    """Single PAM layer — real-valued ablation.

    Same architecture as the complex PAMLayer but:
      - QKV projections are standard nn.Linear
      - Attention is standard Q K^T (no conjugate)
      - State update is standard outer product V @ K^T
      - Decay and GSP logic identical but real-valued
    """
    def __init__(self, dim: int, num_heads: int = 9, head_dim: int = 64,
                 use_rope: bool = True, use_gsp: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.use_gsp = use_gsp
        inner = num_heads * head_dim

        # Fused QKV projection
        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.out_proj = nn.Linear(inner, dim, bias=False)

        # Data-dependent decay
        self.dt_proj = mx.random.normal((num_heads, dim)) * 0.01
        self.dt_bias = mx.ones((num_heads,)) * (-4.0)

        # GSP
        if use_gsp:
            self.gsp_proj = mx.random.normal((num_heads * head_dim, dim)) * 0.01
            self.gsp_bias = mx.ones((num_heads * head_dim,)) * (-3.0)

        # RoPE
        if use_rope:
            self.rope_cos, self.rope_sin = build_rope_freqs(head_dim)

        self.scale = head_dim ** -0.5

    def __call__(self, x):
        """
        x: [B, T, dim]
        Returns: [B, T, dim]
        """
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Fused QKV
        qkv = self.qkv(x)  # [B, T, 3*H*d]
        qkv = mx.reshape(qkv, (B, T, 3, H, d))
        Q = qkv[:, :, 0]  # [B, T, H, d]
        K = qkv[:, :, 1]
        V = qkv[:, :, 2]

        # Transpose to [B, H, T, d]
        Q = mx.transpose(Q, (0, 2, 1, 3))
        K = mx.transpose(K, (0, 2, 1, 3))
        V = mx.transpose(V, (0, 2, 1, 3))

        # RoPE on Q, K
        if self.use_rope:
            Q = apply_rope(Q, self.rope_cos, self.rope_sin)
            K = apply_rope(K, self.rope_cos, self.rope_sin)

        # Data-dependent decay: [B, T, H]
        dt_input = x @ self.dt_proj.T + self.dt_bias  # [B, T, H]
        dt = mx.log(1.0 + mx.exp(dt_input))  # softplus
        gamma = mx.exp(-dt)  # [B, T, H]
        gamma = mx.transpose(gamma, (0, 2, 1))  # [B, H, T]

        # GSP
        if self.use_gsp:
            p = mx.sigmoid(x @ self.gsp_proj.T + self.gsp_bias)  # [B, T, H*d]
            p = mx.reshape(p, (B, T, H, d))
            p = mx.transpose(p, (0, 2, 1, 3))  # [B, H, T, d]
            gamma_eff = mx.expand_dims(gamma, -1) * (1.0 - p) + p  # [B, H, T, d]
            gamma_scalar = mx.mean(gamma_eff, axis=-1)  # [B, H, T]
            V = V * (1.0 - p)
        else:
            gamma_scalar = gamma

        # ── Dual form: O(T^2) parallel computation ──
        log_gamma = mx.log(mx.clip(gamma_scalar, 1e-6, 1.0 - 1e-6))
        log_gamma_cumsum = mx.cumsum(log_gamma, axis=-1)
        log_D = mx.expand_dims(log_gamma_cumsum, -1) - mx.expand_dims(log_gamma_cumsum, -2)
        log_D = mx.clip(log_D, -20.0, 0.0)
        D = mx.exp(log_D)

        # Causal mask
        causal = mx.tril(mx.ones((T, T)))
        D = D * causal

        # Scale Q
        Q_scaled = Q * self.scale

        # Standard attention: W = Q K^T (no conjugate needed)
        W = mx.matmul(Q_scaled, mx.transpose(K, (0, 1, 3, 2)))  # [B, H, T, T]

        # Apply decay
        A = W * D

        # Output: Y = A V
        Y = mx.matmul(A, V)  # [B, H, T, d]

        # Transpose back: [B, T, H, d]
        Y = mx.transpose(Y, (0, 2, 1, 3))
        Y = mx.reshape(Y, (B, T, H * d))

        return self.out_proj(Y)


# ──────────────────────────────────────────────────────────────
# Full Real-PAM Model
# ──────────────────────────────────────────────────────────────

class RPAMBlock(nn.Module):
    """One block: RMSNorm -> SwiGLU + residual -> RMSNorm -> PAM + residual"""
    def __init__(self, dim: int, expand: int = 3, num_heads: int = 9,
                 head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.ffn = SwiGLUBlock(dim, expand)
        self.norm2 = nn.RMSNorm(dim)
        self.pam = PAMLayerReal(dim, num_heads, head_dim)
        self.alpha_cgu = mx.array([1.0])
        self.alpha_pam = mx.array([0.1])

    def __call__(self, x):
        x = x + self.alpha_cgu * self.ffn(self.norm1(x))
        x = x + self.alpha_pam * self.pam(self.norm2(x))
        return x


class RPAMModel(nn.Module):
    """Real-valued PAM language model — ablation of QPAM."""
    def __init__(self, vocab_size: int = 50257, dim: int = 576,
                 num_layers: int = 16, expand: int = 3,
                 num_heads: int = 9, head_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # Real embedding: [vocab, dim]
        scale = math.sqrt(1.0 / dim)
        self.embed = mx.random.normal((vocab_size, dim)) * scale

        self.input_norm = nn.RMSNorm(dim)

        self.blocks = [RPAMBlock(dim, expand, num_heads, head_dim, dropout)
                       for _ in range(num_layers)]

        self.final_proj = nn.Linear(dim, dim, bias=False)
        self.final_norm = nn.RMSNorm(dim)

    def __call__(self, tokens):
        """
        tokens: [B, T] integer token IDs
        Returns: [B, T, vocab_size] real-valued logits
        """
        B, T = tokens.shape

        # Embedding lookup
        x = self.embed[tokens]  # [B, T, dim]

        x = self.input_norm(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_proj(x)
        x = self.final_norm(x)

        # Tied LM head: logits = x @ E^T
        logits = x @ self.embed.T  # [B, T, vocab]

        return logits
