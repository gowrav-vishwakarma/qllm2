"""
QPAM (Quantum Phase-Associative Memory) — MLX implementation for Apple Silicon.

Port of medium-pam-v3 from PyTorch to MLX for fast training on M4 Max.
Core math is identical; representation uses split-real [B,T,D,2] tensors.
"""

import math
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────
# Complex arithmetic helpers (split-real: last dim is 2)
# ──────────────────────────────────────────────────────────────

def cmul(a, b):
    """Complex multiply: (a_r + i a_i)(b_r + i b_i)"""
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return mx.stack([ar * br - ai * bi, ar * bi + ai * br], axis=-1)


def cconj(z):
    """Complex conjugate"""
    return mx.stack([z[..., 0], -z[..., 1]], axis=-1)


def cmag(z):
    """Complex magnitude"""
    return mx.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2 + 1e-8)


def cphase(z):
    """Complex phase angle"""
    return mx.arctan2(z[..., 1], z[..., 0])


# ──────────────────────────────────────────────────────────────
# Phase-preserving primitives
# ──────────────────────────────────────────────────────────────

class ComplexLinear(nn.Module):
    """Complex linear map via block-real GEMM: y = Wz where W, z are complex."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        scale = math.sqrt(2.0 / (in_dim + out_dim))
        self.Wr = mx.random.normal((out_dim, in_dim)) * scale
        self.Wi = mx.random.normal((out_dim, in_dim)) * scale

    def __call__(self, z):
        zr, zi = z[..., 0], z[..., 1]
        yr = zr @ self.Wr.T - zi @ self.Wi.T
        yi = zi @ self.Wr.T + zr @ self.Wi.T
        return mx.stack([yr, yi], axis=-1)


class ComplexNorm(nn.Module):
    """RMS normalization on magnitudes, phase preserved."""
    def __init__(self, dim: int):
        super().__init__()
        self.scale = mx.ones((dim,))

    def __call__(self, z):
        mag = cmag(z)  # [..., dim]
        rms = mx.sqrt(mx.mean(mag ** 2, axis=-1, keepdims=True) + 1e-6)
        normed_mag = mag / rms
        safe_mag = mx.expand_dims(mx.maximum(mag, 1e-6), -1)
        phase = z / safe_mag
        return mx.expand_dims(normed_mag * self.scale, -1) * phase


def mod_relu(z, bias):
    """Phase-preserving activation: modReLU(z) = ReLU(|z| + b) * z/|z|"""
    mag = cmag(z)
    new_mag = mx.maximum(mag + bias, 0.0)
    safe_mag = mx.expand_dims(mx.maximum(mag, 1e-6), -1)
    phase = z / safe_mag
    return mx.expand_dims(new_mag, -1) * phase


class ComplexGatedUnit(nn.Module):
    """SwiGLU-style gating in complex space."""
    def __init__(self, dim: int, expand: int = 3):
        super().__init__()
        hidden = dim * expand
        self.up = ComplexLinear(dim, hidden)
        self.gate = ComplexLinear(dim, hidden)
        self.down = ComplexLinear(hidden, dim)
        self.bias = mx.zeros((hidden,)) - 1.0

    def __call__(self, z):
        up = mod_relu(self.up(z), self.bias)
        gate_out = self.gate(z)
        gate_mag = mx.sigmoid(cmag(gate_out))
        gate_phase = gate_out / mx.expand_dims(mx.maximum(cmag(gate_out), 1e-6), -1)
        gated = cmul(mx.expand_dims(gate_mag, -1) * gate_phase,  up)
        return self.down(gated)


# ──────────────────────────────────────────────────────────────
# Complex RoPE
# ──────────────────────────────────────────────────────────────

def build_rope_freqs(dim: int, max_len: int = 8192, base: float = 10000.0):
    """Precompute RoPE frequencies as complex unit vectors."""
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(max_len, dtype=mx.float32)
    angles = mx.outer(t, freqs)  # [T, dim/2]
    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)
    # Repeat to match dim: each pair of dims gets same freq
    cos_a = mx.repeat(cos_a, 2, axis=-1)[:, :dim]
    sin_a = mx.repeat(sin_a, 2, axis=-1)[:, :dim]
    return cos_a, sin_a


def apply_complex_rope(z, cos_freqs, sin_freqs, offset: int = 0):
    """Apply complex RoPE: multiply each complex element by e^{i*m*theta}."""
    T = z.shape[-3]
    cos_f = cos_freqs[offset:offset + T]  # [T, d]
    sin_f = sin_freqs[offset:offset + T]
    # z is [..., T, d, 2], cos/sin are [T, d]
    zr, zi = z[..., 0], z[..., 1]
    yr = zr * cos_f - zi * sin_f
    yi = zr * sin_f + zi * cos_f
    return mx.stack([yr, yi], axis=-1)


# ──────────────────────────────────────────────────────────────
# Phase-Associative Memory (PAM) Layer
# ──────────────────────────────────────────────────────────────

class PAMLayer(nn.Module):
    """Single PAM layer with GSP, complex RoPE, fused QKV."""
    def __init__(self, dim: int, num_heads: int = 6, head_dim: int = 64,
                 use_rope: bool = True, use_gsp: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.use_gsp = use_gsp
        inner = num_heads * head_dim

        # Fused QKV projection
        self.qkv = ComplexLinear(dim, 3 * inner)
        self.out_proj = ComplexLinear(inner, dim)

        # Data-dependent decay
        self.dt_proj = mx.random.normal((num_heads, 2 * dim)) * 0.01
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
        x: [B, T, dim, 2] (complex)
        Returns: [B, T, dim, 2]
        """
        B, T, D, _ = x.shape
        H, d = self.num_heads, self.head_dim

        # Fused QKV
        qkv = self.qkv(x)  # [B, T, 3*H*d, 2]
        qkv = mx.reshape(qkv, (B, T, 3, H, d, 2))
        Q = qkv[:, :, 0]  # [B, T, H, d, 2]
        K = qkv[:, :, 1]
        V = qkv[:, :, 2]

        # Transpose to [B, H, T, d, 2]
        Q = mx.transpose(Q, (0, 2, 1, 3, 4))
        K = mx.transpose(K, (0, 2, 1, 3, 4))
        V = mx.transpose(V, (0, 2, 1, 3, 4))

        # Complex RoPE on Q, K
        if self.use_rope:
            Q = apply_complex_rope(Q, self.rope_cos, self.rope_sin)
            K = apply_complex_rope(K, self.rope_cos, self.rope_sin)

        # Data-dependent decay: [B, T, H]
        x_flat = mx.reshape(x, (B, T, D * 2))  # concat real/imag
        dt_input = x_flat @ self.dt_proj.T + self.dt_bias  # [B, T, H]
        dt = mx.log(1.0 + mx.exp(dt_input))  # softplus
        gamma = mx.exp(-dt)  # [B, T, H]
        # Transpose to [B, H, T]
        gamma = mx.transpose(gamma, (0, 2, 1))

        # GSP
        if self.use_gsp:
            x_mag = cmag(x)  # [B, T, D]
            p = mx.sigmoid(x_mag @ self.gsp_proj.T + self.gsp_bias)  # [B, T, H*d]
            p = mx.reshape(p, (B, T, H, d))
            p = mx.transpose(p, (0, 2, 1, 3))  # [B, H, T, d]
            # Modify gamma: gamma_eff = exp(-dt)(1-p) + p  per dimension
            gamma_eff = mx.expand_dims(gamma, -1) * (1.0 - p) + p  # [B, H, T, d]
            # For dual form, use scalar decay (average over d)
            gamma_scalar = mx.mean(gamma_eff, axis=-1)  # [B, H, T]
            # Modulate V: V' = V * (1 - p)
            V_prime_scale = mx.expand_dims(1.0 - p, -1)  # [B, H, T, d, 1]
            V = V * V_prime_scale
        else:
            gamma_scalar = gamma  # already [B, H, T]

        # ── Dual form: O(T²) parallel computation ──
        # Decay matrix D[t,i] = prod_{j=i+1}^{t} gamma_j
        log_gamma = mx.log(mx.clip(gamma_scalar, 1e-4, 1.0 - 1e-4))  # [B, H, T]
        log_gamma_cumsum = mx.cumsum(log_gamma, axis=-1)  # [B, H, T]
        # D[t,i] = exp(cumsum[t] - cumsum[i])
        log_D = mx.expand_dims(log_gamma_cumsum, -1) - mx.expand_dims(log_gamma_cumsum, -2)  # [B, H, T, T]
        # Clamp to prevent overflow/underflow in exp
        log_D = mx.clip(log_D, -20.0, 0.0)
        D = mx.exp(log_D)

        # Causal mask
        causal = mx.tril(mx.ones((T, T)))
        D = D * causal

        # Scale Q
        Q_scaled = Q * self.scale

        # Complex attention: W = Q̃ K*^T
        Qr, Qi = Q_scaled[..., 0], Q_scaled[..., 1]
        Kr, Ki = K[..., 0], K[..., 1]
        # W = (Qr Kr^T + Qi Ki^T) + i(Qi Kr^T - Qr Ki^T)  [conjugate of K]
        Wr = mx.matmul(Qr, mx.transpose(Kr, (0, 1, 3, 2))) + mx.matmul(Qi, mx.transpose(Ki, (0, 1, 3, 2)))
        Wi = mx.matmul(Qi, mx.transpose(Kr, (0, 1, 3, 2))) - mx.matmul(Qr, mx.transpose(Ki, (0, 1, 3, 2)))

        # Apply decay
        Ar = Wr * D
        Ai = Wi * D

        # Output: Y = A V'
        Vr, Vi = V[..., 0], V[..., 1]
        Yr = mx.matmul(Ar, Vr) - mx.matmul(Ai, Vi)
        Yi = mx.matmul(Ar, Vi) + mx.matmul(Ai, Vr)
        Y = mx.stack([Yr, Yi], axis=-1)  # [B, H, T, d, 2]

        # Transpose back: [B, T, H, d, 2]
        Y = mx.transpose(Y, (0, 2, 1, 3, 4))
        Y = mx.reshape(Y, (B, T, H * d, 2))

        return self.out_proj(Y)


# ──────────────────────────────────────────────────────────────
# Full QPAM Model
# ──────────────────────────────────────────────────────────────

class QPAMBlock(nn.Module):
    """One block: ComplexNorm -> CGU + residual -> ComplexNorm -> PAM + residual"""
    def __init__(self, dim: int, expand: int = 3, num_heads: int = 6,
                 head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm1 = ComplexNorm(dim)
        self.cgu = ComplexGatedUnit(dim, expand)
        self.norm2 = ComplexNorm(dim)
        self.pam = PAMLayer(dim, num_heads, head_dim)
        self.alpha_cgu = mx.array([1.0])
        self.alpha_pam = mx.array([0.1])

    def __call__(self, z):
        z = z + self.alpha_cgu * self.cgu(self.norm1(z))
        z = z + self.alpha_pam * self.pam(self.norm2(z))
        return z


class QPAMModel(nn.Module):
    """Full QPAM language model."""
    def __init__(self, vocab_size: int = 50257, dim: int = 384,
                 num_layers: int = 16, expand: int = 3,
                 num_heads: int = 6, head_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # Complex embedding: [vocab, dim, 2]
        scale = math.sqrt(1.0 / dim)
        self.embed_r = mx.random.normal((vocab_size, dim)) * scale
        self.embed_i = mx.random.normal((vocab_size, dim)) * scale

        self.input_norm = ComplexNorm(dim)

        self.blocks = [QPAMBlock(dim, expand, num_heads, head_dim, dropout)
                       for _ in range(num_layers)]

        self.final_proj = ComplexLinear(dim, dim)
        self.final_norm = ComplexNorm(dim)

    def __call__(self, tokens):
        """
        tokens: [B, T] integer token IDs
        Returns: [B, T, vocab_size] real-valued logits
        """
        B, T = tokens.shape

        # Complex embedding lookup
        er = self.embed_r[tokens]  # [B, T, dim]
        ei = self.embed_i[tokens]
        z = mx.stack([er, ei], axis=-1)  # [B, T, dim, 2]

        z = self.input_norm(z)

        for block in self.blocks:
            z = block(z)

        z = self.final_proj(z)
        z = self.final_norm(z)

        # Tied complex LM head: logits = Re(z · conj(E))
        zr, zi = z[..., 0], z[..., 1]  # [B, T, dim]
        logits = zr @ self.embed_r.T + zi @ self.embed_i.T  # [B, T, vocab]

        return logits
