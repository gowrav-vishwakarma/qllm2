#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase2D: GPU-friendly complex/phase representation using 2D real vectors

Core idea: represent complex numbers as 2D real vectors (real, imag).
Multiplication by i (iota) is just a 90° rotation: J @ v where J = [[0, -1], [1, 0]].
All operations reduce to GEMM/matmul - no sin/cos in the hot path.

Shapes:
- Phase2D tensor: [..., 2] where last dim is (real, imag)
- Or equivalently: [..., dim, 2] for dim-dimensional phase vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from dataclasses import dataclass


# =============================================================================
# Core Phase2D Operations (all GEMM-friendly, no trig)
# =============================================================================

@torch.jit.script
def phase2d_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication: (a_r + i*a_i) * (b_r + i*b_i)
    Input shapes: [..., 2] where last dim is (real, imag)
    Output: [..., 2]
    """
    a_real, a_imag = a[..., 0], a[..., 1]
    b_real, b_imag = b[..., 0], b[..., 1]
    
    out_real = a_real * b_real - a_imag * b_imag
    out_imag = a_real * b_imag + a_imag * b_real
    
    return torch.stack([out_real, out_imag], dim=-1)


@torch.jit.script
def phase2d_conjugate(x: torch.Tensor) -> torch.Tensor:
    """Complex conjugate: (r, i) -> (r, -i)"""
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


@torch.jit.script
def phase2d_magnitude_sq(x: torch.Tensor) -> torch.Tensor:
    """Squared magnitude: |z|^2 = r^2 + i^2"""
    return x[..., 0] ** 2 + x[..., 1] ** 2


@torch.jit.script
def phase2d_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Magnitude: |z| = sqrt(r^2 + i^2)"""
    return torch.sqrt(phase2d_magnitude_sq(x) + 1e-8)


@torch.jit.script
def phase2d_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize to unit circle: z / |z|"""
    mag = phase2d_magnitude(x).unsqueeze(-1)
    return x / mag


@torch.jit.script
def phase2d_apply_iota(x: torch.Tensor) -> torch.Tensor:
    """
    Multiply by i (90° rotation): i * (r + i*im) = -im + i*r
    This is just J @ v where J = [[0, -1], [1, 0]]
    """
    return torch.stack([-x[..., 1], x[..., 0]], dim=-1)


@torch.jit.script
def phase2d_rotate(x: torch.Tensor, angle_real: torch.Tensor, angle_imag: torch.Tensor) -> torch.Tensor:
    """
    Rotate by a complex phase (angle_real + i*angle_imag).
    For unit rotation, use normalized (cos, sin) pair.
    This is just complex multiplication.
    """
    x_real, x_imag = x[..., 0], x[..., 1]
    
    out_real = x_real * angle_real - x_imag * angle_imag
    out_imag = x_real * angle_imag + x_imag * angle_real
    
    return torch.stack([out_real, out_imag], dim=-1)


@torch.jit.script
def phase2d_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complex dot product: sum(a * conj(b))
    Returns [..., 2] complex result
    """
    b_conj = phase2d_conjugate(b)
    prod = phase2d_multiply(a, b_conj)
    return prod.sum(dim=-2)  # sum over the vector dimension


@torch.jit.script
def phase2d_coherence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Coherence between two phase vectors: Re(a · conj(b)) / (|a| * |b|)
    Returns real scalar per batch element.
    No trig - just normalized dot product.
    """
    dot_prod = phase2d_multiply(a, phase2d_conjugate(b))
    dot_real = dot_prod[..., 0].sum(dim=-1)  # real part of dot product
    
    mag_a = phase2d_magnitude(a).sum(dim=-1)
    mag_b = phase2d_magnitude(b).sum(dim=-1)
    
    return dot_real / (mag_a * mag_b + 1e-8)


# =============================================================================
# Phase2D Conversion Utilities
# =============================================================================

def phase2d_from_real(x: torch.Tensor, mode: str = 'split') -> torch.Tensor:
    """
    Convert real tensor to Phase2D.
    
    Args:
        x: Real tensor of shape [..., dim]
        mode: 
            'split' - split dim in half: first half = real, second half = imag
            'zero_imag' - real = x, imag = 0
            'learned' - requires external projection (use Phase2DEmbed instead)
    
    Returns:
        Phase2D tensor [..., dim//2, 2] or [..., dim, 2]
    """
    if mode == 'split':
        dim = x.shape[-1]
        assert dim % 2 == 0, f"Dim must be even for split mode, got {dim}"
        real = x[..., :dim // 2]
        imag = x[..., dim // 2:]
        return torch.stack([real, imag], dim=-1)
    elif mode == 'zero_imag':
        imag = torch.zeros_like(x)
        return torch.stack([x, imag], dim=-1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def phase2d_to_real(x: torch.Tensor, mode: str = 'concat') -> torch.Tensor:
    """
    Convert Phase2D tensor back to real.
    
    Args:
        x: Phase2D tensor [..., dim, 2]
        mode:
            'concat' - concatenate real and imag: [..., dim*2]
            'real_only' - return only real part: [..., dim]
            'magnitude' - return magnitude: [..., dim]
    
    Returns:
        Real tensor
    """
    if mode == 'concat':
        return torch.cat([x[..., 0], x[..., 1]], dim=-1)
    elif mode == 'real_only':
        return x[..., 0]
    elif mode == 'magnitude':
        return phase2d_magnitude(x)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# IotaBlock: 2x2 block parameterization for rotations (Cayley transform)
# =============================================================================

class IotaBlock(nn.Module):
    """
    Learnable rotation/unitary transformation using Cayley transform.
    
    Instead of learning angles (which require sin/cos), we learn a skew-symmetric
    parameter and compute rotation via Cayley transform.
    
    For Cayley: cos = (1 - a^2) / (1 + a^2), sin = 2a / (1 + a^2)
    
    This is an orthogonal matrix (rotation) without any trig.
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim: dimension of the phase space
        """
        super().__init__()
        self.dim = dim
        
        # Learnable skew-symmetric parameter (one per dimension)
        # Initialize small for near-identity rotations
        self.skew_params = nn.Parameter(torch.randn(dim) * 0.01)
    
    def get_rotation_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotation matrix components using Cayley transform.
        Returns (cos_like, sin_like) for rotation.
        
        For Cayley: cos = (1 - a^2) / (1 + a^2), sin = 2a / (1 + a^2)
        """
        a = self.skew_params
        a_sq = a * a
        denom = 1.0 + a_sq
        
        cos_like = (1.0 - a_sq) / denom  # [dim]
        sin_like = (2.0 * a) / denom      # [dim]
        
        return cos_like, sin_like
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to Phase2D input.
        
        Args:
            x: [..., dim, 2] Phase2D tensor
        
        Returns:
            [..., dim, 2] rotated Phase2D tensor
        """
        # Get rotation components
        cos_like, sin_like = self.get_rotation_components()
        
        # Reshape for broadcasting to [..., dim]
        # cos_like and sin_like are [dim], need to match x's [..., dim, 2]
        x_real = x[..., 0]  # [..., dim]
        x_imag = x[..., 1]  # [..., dim]
        
        # Apply rotation: complex multiplication by (cos + i*sin)
        out_real = x_real * cos_like - x_imag * sin_like
        out_imag = x_real * sin_like + x_imag * cos_like
        
        return torch.stack([out_real, out_imag], dim=-1)


# =============================================================================
# Phase2D Embedding Layer
# =============================================================================

class Phase2DEmbed(nn.Module):
    """
    Embed tokens into Phase2D space.
    
    Projects token embeddings into complex phase space with learnable
    real and imaginary components.
    """
    
    def __init__(self, vocab_size: int, dim: int, padding_idx: Optional[int] = None):
        """
        Args:
            vocab_size: vocabulary size
            dim: phase dimension (output will be [batch, seq, dim, 2])
            padding_idx: padding token index
        """
        super().__init__()
        self.dim = dim
        
        # Separate embeddings for real and imaginary parts
        self.embed_real = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.embed_imag = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        
        # Initialize: real part standard, imag part smaller
        nn.init.normal_(self.embed_real.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embed_imag.weight, mean=0.0, std=0.01)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq] token indices
        
        Returns:
            [batch, seq, dim, 2] Phase2D embeddings
        """
        real = self.embed_real(token_ids)  # [batch, seq, dim]
        imag = self.embed_imag(token_ids)  # [batch, seq, dim]
        
        return torch.stack([real, imag], dim=-1)  # [batch, seq, dim, 2]


# =============================================================================
# Phase2D Linear Layer (complex-valued linear)
# =============================================================================

class Phase2DLinear(nn.Module):
    """
    Linear layer that operates on Phase2D tensors.
    
    Implements complex linear: y = W @ x where W is complex.
    W = W_real + i * W_imag
    """
    
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Complex weight: W = W_real + i * W_imag
        self.weight_real = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_dim))
            self.bias_imag = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_dim, 2] Phase2D input
        
        Returns:
            [..., out_dim, 2] Phase2D output
        """
        x_real, x_imag = x[..., 0], x[..., 1]
        
        # Complex matrix multiplication:
        # (W_r + i*W_i) @ (x_r + i*x_i) = (W_r@x_r - W_i@x_i) + i*(W_r@x_i + W_i@x_r)
        out_real = F.linear(x_real, self.weight_real) - F.linear(x_imag, self.weight_imag)
        out_imag = F.linear(x_real, self.weight_imag) + F.linear(x_imag, self.weight_real)
        
        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag
        
        return torch.stack([out_real, out_imag], dim=-1)


# =============================================================================
# Phase2D Layer Norm (normalize magnitude, preserve phase)
# =============================================================================

class Phase2DLayerNorm(nn.Module):
    """
    Layer normalization for Phase2D tensors.
    Normalizes by magnitude while preserving phase relationships.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # Learnable scale (applied to magnitude)
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., dim, 2] Phase2D tensor
        
        Returns:
            [..., dim, 2] normalized Phase2D tensor
        """
        # Compute magnitude
        mag = phase2d_magnitude(x)  # [..., dim]
        
        # Compute mean and variance of magnitude
        mean = mag.mean(dim=-1, keepdim=True)
        var = ((mag - mean) ** 2).mean(dim=-1, keepdim=True)
        
        # Normalize magnitude
        mag_norm = (mag - mean) / torch.sqrt(var + self.eps)
        
        # Scale normalized magnitude
        mag_scaled = mag_norm * self.scale
        
        # Apply to phase2d: keep direction, change magnitude
        direction = phase2d_normalize(x)
        
        return direction * mag_scaled.unsqueeze(-1)


# =============================================================================
# Convenience Wrapper
# =============================================================================

@dataclass
class Phase2D:
    """
    Convenience wrapper for Phase2D tensors with operator overloading.
    Use for cleaner code when prototyping.
    """
    data: torch.Tensor  # [..., 2] or [..., dim, 2]
    
    @property
    def real(self) -> torch.Tensor:
        return self.data[..., 0]
    
    @property
    def imag(self) -> torch.Tensor:
        return self.data[..., 1]
    
    @property
    def magnitude(self) -> torch.Tensor:
        return phase2d_magnitude(self.data)
    
    def conjugate(self) -> 'Phase2D':
        return Phase2D(phase2d_conjugate(self.data))
    
    def normalize(self) -> 'Phase2D':
        return Phase2D(phase2d_normalize(self.data))
    
    def __mul__(self, other: 'Phase2D') -> 'Phase2D':
        return Phase2D(phase2d_multiply(self.data, other.data))
    
    def __add__(self, other: 'Phase2D') -> 'Phase2D':
        return Phase2D(self.data + other.data)
    
    def __sub__(self, other: 'Phase2D') -> 'Phase2D':
        return Phase2D(self.data - other.data)
    
    def to_real(self, mode: str = 'concat') -> torch.Tensor:
        return phase2d_to_real(self.data, mode)
    
    @staticmethod
    def from_real(x: torch.Tensor, mode: str = 'split') -> 'Phase2D':
        return Phase2D(phase2d_from_real(x, mode))
    
    def coherence(self, other: 'Phase2D') -> torch.Tensor:
        return phase2d_coherence(self.data, other.data)
