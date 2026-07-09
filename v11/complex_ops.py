"""
V11 complex primitives (vendored from V7).

Split-real complex tensors: [..., dim, 2] — never torch.complex64/128.
Used by V11PAMLayer, V11Block, V11LM, and duplex model heads.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from v11.triton_kernels import (
    fused_cgu_gate,
    fused_complex_norm,
    fused_mod_relu,
    fused_mod_swish,
)


# ── Complex Arithmetic (split-real: [..., dim, 2]) ───────────────────────────

@torch.jit.script
def cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(a_r + i·a_i)(b_r + i·b_i)"""
    return torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
    ], dim=-1)


@torch.jit.script
def cconj(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


@torch.jit.script
def cabs(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x[..., 0].square() + x[..., 1].square() + 1e-8)


@torch.jit.script
def cnormalize(x: torch.Tensor) -> torch.Tensor:
    return x / cabs(x).unsqueeze(-1)


@torch.jit.script
def to_real_concat(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[..., 0], x[..., 1]], dim=-1)


# ── Complex Modules ──────────────────────────────────────────────────────────

class ModReLU(nn.Module):
    """Phase-preserving activation: threshold on magnitude, phase untouched."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), -0.1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return fused_mod_relu(z, self.bias)


class ModSwish(nn.Module):
    """Smooth phase-preserving activation: Swish on magnitude, phase untouched."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return fused_mod_swish(z, self.bias, self.beta)


class PhaseModulatedActivation(nn.Module):
    """Activation that couples magnitude and phase."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))
        self.phase_alpha = nn.Parameter(torch.zeros(dim))
        self.phase_beta = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        magnitude = cabs(z)
        activated_magnitude = magnitude * torch.sigmoid(self.beta * magnitude + self.bias)
        # preserve direction on the unit circle before scaling by activated magnitude
        phase = z / (magnitude.unsqueeze(-1) + 1e-8)
        rotation_angle = self.phase_alpha * magnitude + self.phase_beta
        rotation = torch.stack([rotation_angle.cos(), rotation_angle.sin()], dim=-1)
        phase = cmul(phase, rotation)
        return phase * activated_magnitude.unsqueeze(-1)


class ComplexLinear(nn.Module):
    """Complex linear via split real/imag matmuls with orthogonal init."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        init_scale = (2 / (in_dim + out_dim)) ** 0.5
        self.weight_real = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_imag = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.orthogonal_(self.weight_real, gain=init_scale)
        nn.init.orthogonal_(self.weight_imag, gain=init_scale)
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_dim))
            self.bias_imag = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real_in, imag_in = x[..., 0], x[..., 1]
        real_out = F.linear(real_in, self.weight_real) - F.linear(imag_in, self.weight_imag)
        imag_out = F.linear(real_in, self.weight_imag) + F.linear(imag_in, self.weight_real)
        if self.bias_real is not None:
            real_out = real_out + self.bias_real
            imag_out = imag_out + self.bias_imag
        return torch.stack([real_out, imag_out], dim=-1)


class ComplexNorm(nn.Module):
    """RMSNorm for complex: normalize magnitude, preserve phase."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return fused_complex_norm(z, self.scale, self.eps)


def _build_activation(name: str, dim: int) -> nn.Module:
    if name == 'swish':
        return ModSwish(dim)
    elif name == 'phase_mod':
        return PhaseModulatedActivation(dim)
    return ModReLU(dim)


class ComplexGatedUnit(nn.Module):
    """SwiGLU-style complex gating: magnitude gates how much, phase gates rotation."""

    def __init__(self, dim: int, expand: int = 3, activation: str = 'modrelu'):
        super().__init__()
        hidden_dim = dim * expand
        self.gate_proj = ComplexLinear(dim, hidden_dim, bias=False)
        self.up_proj = ComplexLinear(dim, hidden_dim, bias=False)
        self.down_proj = ComplexLinear(hidden_dim, dim, bias=False)
        self.act = _build_activation(activation, hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(z)
        up = self.act(self.up_proj(z))
        gated = fused_cgu_gate(gate, up)
        return self.down_proj(gated)


class ComplexEmbed(nn.Module):
    """Embed tokens into complex space: real + imaginary components."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.dim = dim
        self.embed_real = nn.Embedding(vocab_size, dim)
        self.embed_imag = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed_real.weight, std=0.02)
        nn.init.normal_(self.embed_imag.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.embed_real(ids), self.embed_imag(ids)], dim=-1)


class ComplexPosEmbed(nn.Module):
    """Learned absolute position embed added to token embed before the stack."""

    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, z: torch.Tensor, step_offset: int = 0) -> torch.Tensor:
        # z: [B, T, dim, 2]
        seq_len = z.shape[1]
        position_end = step_offset + seq_len
        if position_end > self.max_seq_len:
            raise ValueError(
                f"Position range [{step_offset}, {position_end}) exceeds max_seq_len "
                f"{self.max_seq_len}"
            )
        position_ids = torch.arange(step_offset, position_end, device=z.device)
        position_embed = self.pos_embed(position_ids)  # [T, dim]
        return z + position_embed.unsqueeze(0).unsqueeze(-1)


def build_rope_cache(max_len: int, head_dim: int) -> torch.Tensor:
    """Complex RoPE: e^{i·m·theta_k} for positions m and frequency bands k."""
    inverse_freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * inverse_freqs.unsqueeze(0)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)
