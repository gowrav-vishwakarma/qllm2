"""
Complex algebraic primitives for V6.

Forked from V5 unchanged -- these are the non-negotiable phase-safe building
blocks. Every operation preserves the algebraic structure of complex numbers.

Key principle: NEVER apply real-valued activations to complex data.
Use modReLU (phase-preserving) instead of GELU/ReLU on real parts.

Representation: tensors of shape [..., dim, 2] where last dim is (real, imag).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..init import InitStrategy


# ---------------------------------------------------------------------------
# Elementwise complex arithmetic (all JIT-scriptable)
# ---------------------------------------------------------------------------

@torch.jit.script
def cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex multiply: (a_r+i*a_i)(b_r+i*b_i)."""
    return torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
    ], dim=-1)


@torch.jit.script
def cconj(x: torch.Tensor) -> torch.Tensor:
    """Complex conjugate: (r, i) -> (r, -i)."""
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


@torch.jit.script
def cabs(x: torch.Tensor) -> torch.Tensor:
    """Magnitude |z| = sqrt(r^2 + i^2), shape [..., dim]."""
    return torch.sqrt(x[..., 0].square() + x[..., 1].square() + 1e-8)


@torch.jit.script
def cabs2(x: torch.Tensor) -> torch.Tensor:
    """Squared magnitude |z|^2, shape [..., dim]."""
    return x[..., 0].square() + x[..., 1].square()


@torch.jit.script
def cnormalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize to unit magnitude, preserving phase: z / |z|."""
    mag = cabs(x).unsqueeze(-1)
    return x / mag


@torch.jit.script
def creal_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Re(a * conj(b)) summed over feature dim. Returns [...] scalar."""
    return (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(dim=-1)


class ModReLU(nn.Module):
    """
    Phase-preserving activation: threshold on magnitude, phase untouched.

        modReLU(z) = ReLU(|z| + b) * z/|z|      if |z| + b > 0
                     0                            otherwise
    """

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), -0.1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = cabs(z)
        activated_mag = F.relu(mag + self.bias)
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * activated_mag.unsqueeze(-1)


class ComplexLinear(nn.Module):
    """Complex-valued linear layer: y = W @ x + b, all complex."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if initializer is not None:
            wr, wi = initializer.init_complex_linear(out_dim, in_dim)
            self.weight_real = nn.Parameter(wr)
            self.weight_imag = nn.Parameter(wi)
        else:
            scale = (2 / (in_dim + out_dim)) ** 0.5
            self.weight_real = nn.Parameter(torch.randn(out_dim, in_dim) * scale)
            self.weight_imag = nn.Parameter(torch.randn(out_dim, in_dim) * scale)

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_dim))
            self.bias_imag = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = torch.cat([x[..., 0], x[..., 1]], dim=-1)
        W_block = torch.cat([
            torch.cat([self.weight_real, -self.weight_imag], dim=1),
            torch.cat([self.weight_imag,  self.weight_real], dim=1),
        ], dim=0)
        bias = None
        if self.bias_real is not None:
            bias = torch.cat([self.bias_real, self.bias_imag])
        y_flat = F.linear(x_flat, W_block, bias)
        return torch.stack([y_flat[..., :self.out_dim], y_flat[..., self.out_dim:]], dim=-1)


class ComplexNorm(nn.Module):
    """Magnitude normalization, phase preserved. Like RMSNorm for complex."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = cabs(z)
        rms = torch.sqrt(mag.square().mean(dim=-1, keepdim=True) + self.eps)
        mag_normed = mag / rms
        mag_scaled = mag_normed * self.scale
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * mag_scaled.unsqueeze(-1)


class ComplexGatedUnit(nn.Module):
    """
    SwiGLU-style complex gating block.

    Gate magnitude sigma(|W_g z|) selects HOW MUCH, gate phase selects
    WHAT ROTATION. Up-projection + modReLU provides nonlinearity.
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        hidden = dim * expand
        self.gate_proj = ComplexLinear(dim, hidden, bias=False, initializer=initializer)
        self.up_proj = ComplexLinear(dim, hidden, bias=False, initializer=initializer)
        self.down_proj = ComplexLinear(hidden, dim, bias=False, initializer=initializer)
        self.act = ModReLU(hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(z)
        gate_mag = torch.sigmoid(cabs(gate))
        gate_phase = gate / (cabs(gate).unsqueeze(-1) + 1e-8)
        up = self.act(self.up_proj(z))
        gated = cmul(gate_phase, up) * gate_mag.unsqueeze(-1)
        return self.down_proj(gated)


class ComplexEmbed(nn.Module):
    """Embed tokens into complex space with learned real and imaginary components."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        padding_idx: Optional[int] = None,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.dim = dim
        self.embed_real = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.embed_imag = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)

        if initializer is not None:
            r, i = initializer.init_embedding(vocab_size, dim)
            with torch.no_grad():
                self.embed_real.weight.copy_(r)
                self.embed_imag.weight.copy_(i)
        else:
            nn.init.normal_(self.embed_real.weight, std=0.02)
            nn.init.normal_(self.embed_imag.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        r = self.embed_real(token_ids)
        i = self.embed_imag(token_ids)
        return torch.stack([r, i], dim=-1)


def build_rope_cache(max_len: int, head_dim: int) -> torch.Tensor:
    """Precompute complex RoPE: e^{i*m*theta_k} for m=0..max_len-1, k=0..head_dim-1."""
    freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [max_len, head_dim]
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # [max_len, head_dim, 2]


def to_real(z: torch.Tensor, mode: str = 'concat') -> torch.Tensor:
    """Convert [..., dim, 2] complex to real tensor for output heads."""
    if mode == 'concat':
        return torch.cat([z[..., 0], z[..., 1]], dim=-1)
    elif mode == 'magnitude':
        return cabs(z)
    elif mode == 'real':
        return z[..., 0]
    raise ValueError(f"Unknown mode: {mode}")
