"""
RPAM (Real-valued PAM) — PyTorch implementation.
Real-valued ablation: same architecture as QPAM but with standard dot products,
real-valued outer products, and SwiGLU instead of ComplexGatedUnit.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLUBlock(nn.Module):
    def __init__(self, dim: int, expand: int = 3):
        super().__init__()
        hidden = dim * expand
        self.up = nn.Linear(dim, hidden, bias=False)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class PAMLayerReal(nn.Module):
    """Real-valued PAM layer. Standard dot-product attention with decay."""
    def __init__(self, dim: int, num_heads: int = 9, head_dim: int = 64,
                 use_rope: bool = True, use_gsp: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.use_gsp = use_gsp
        inner = num_heads * head_dim

        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.out_proj = nn.Linear(inner, dim, bias=False)

        # Data-dependent decay
        self.dt_proj = nn.Linear(dim, num_heads, bias=True)
        nn.init.constant_(self.dt_proj.bias, -4.0)
        nn.init.normal_(self.dt_proj.weight, std=0.01)

        # GSP
        if use_gsp:
            self.gsp_proj = nn.Linear(dim, num_heads * head_dim, bias=True)
            nn.init.constant_(self.gsp_proj.bias, -3.0)
            nn.init.normal_(self.gsp_proj.weight, std=0.01)

        # RoPE
        if use_rope:
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            self.register_buffer('rope_inv_freq', inv_freq)

        self.scale = head_dim ** -0.5

    def _apply_rope(self, x, offset: int = 0):
        """x: [B, H, T, d]"""
        B, H, T, d = x.shape
        positions = torch.arange(offset, offset + T, device=x.device, dtype=torch.float32)
        freqs = torch.outer(positions, self.rope_inv_freq)  # [T, d/2]
        cos_f = torch.cos(freqs)  # [T, d/2]
        sin_f = torch.sin(freqs)
        cos_f = cos_f.unsqueeze(0).unsqueeze(0)  # [1, 1, T, d/2]
        sin_f = sin_f.unsqueeze(0).unsqueeze(0)
        x1 = x[..., 0::2]  # even dims
        x2 = x[..., 1::2]  # odd dims
        y1 = x1 * cos_f - x2 * sin_f
        y2 = x1 * sin_f + x2 * cos_f
        out = torch.stack([y1, y2], dim=-1).flatten(-2)  # interleave
        return out

    def forward(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        qkv = self.qkv(x)  # [B, T, 3*H*d]
        qkv = qkv.view(B, T, 3, H, d)
        Q = qkv[:, :, 0].transpose(1, 2)  # [B, H, T, d]
        K = qkv[:, :, 1].transpose(1, 2)
        V = qkv[:, :, 2].transpose(1, 2)

        if self.use_rope:
            Q = self._apply_rope(Q)
            K = self._apply_rope(K)

        # Decay
        dt = F.softplus(self.dt_proj(x))  # [B, T, H]
        gamma = torch.exp(-dt).transpose(1, 2)  # [B, H, T]

        # GSP
        if self.use_gsp:
            p = torch.sigmoid(self.gsp_proj(x))  # [B, T, H*d]
            p = p.view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]
            gamma_eff = gamma.unsqueeze(-1) * (1.0 - p) + p  # [B, H, T, d]
            gamma_scalar = gamma_eff.mean(dim=-1)  # [B, H, T]
            V = V * (1.0 - p)
        else:
            gamma_scalar = gamma

        # Dual form
        log_gamma = torch.log(gamma_scalar.clamp(1e-6, 1.0 - 1e-6))
        log_gamma_cumsum = torch.cumsum(log_gamma, dim=-1)
        log_D = log_gamma_cumsum.unsqueeze(-1) - log_gamma_cumsum.unsqueeze(-2)
        log_D = log_D.clamp(-20.0, 0.0)
        D = torch.exp(log_D)
        causal = torch.tril(torch.ones(T, T, device=x.device))
        D = D * causal

        Q_scaled = Q * self.scale
        W = torch.matmul(Q_scaled, K.transpose(-2, -1))  # [B, H, T, T]
        A = W * D
        Y = torch.matmul(A, V)  # [B, H, T, d]

        Y = Y.transpose(1, 2).contiguous().view(B, T, H * d)
        return self.out_proj(Y)


class RPAMBlock(nn.Module):
    def __init__(self, dim: int, expand: int = 3, num_heads: int = 9,
                 head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.ffn = SwiGLUBlock(dim, expand)
        self.norm2 = nn.RMSNorm(dim)
        self.pam = PAMLayerReal(dim, num_heads, head_dim)
        self.alpha_cgu = nn.Parameter(torch.tensor(1.0))
        self.alpha_pam = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        x = x + self.alpha_cgu * self.ffn(self.norm1(x))
        x = x + self.alpha_pam * self.pam(self.norm2(x))
        return x


class RPAMModel(nn.Module):
    def __init__(self, vocab_size: int = 50257, dim: int = 576,
                 num_layers: int = 16, expand: int = 3,
                 num_heads: int = 9, head_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed.weight, std=math.sqrt(1.0 / dim))

        self.input_norm = nn.RMSNorm(dim)
        self.blocks = nn.ModuleList([
            RPAMBlock(dim, expand, num_heads, head_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_proj = nn.Linear(dim, dim, bias=False)
        self.final_norm = nn.RMSNorm(dim)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_proj(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight)  # tied
        return logits