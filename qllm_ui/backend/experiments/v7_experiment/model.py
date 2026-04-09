import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        scale = 1.0 / (in_features ** 0.5)
        self.weight_r = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.weight_i = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.bias_r = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.bias_i = nn.Parameter(torch.zeros(out_features)) if bias else None
    def forward(self, x):
        r, i = x[..., 0], x[..., 1]
        out_r = F.linear(r, self.weight_r) - F.linear(i, self.weight_i)
        out_i = F.linear(r, self.weight_i) + F.linear(i, self.weight_r)
        if self.bias_r is not None:
            out_r = out_r + self.bias_r
            out_i = out_i + self.bias_i
        return torch.stack([out_r, out_i], dim=-1)

class ComplexNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = (x * x).mean(dim=-2, keepdim=True).clamp(min=self.eps).sqrt()
        return x * (self.gamma.unsqueeze(-1) / rms)

class ComplexEmbed(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embed_r = nn.Embedding(vocab_size, dim)
        self.embed_i = nn.Embedding(vocab_size, dim)
    def forward(self, input_ids):
        return torch.stack([self.embed_r(input_ids), self.embed_i(input_ids)], dim=-1)

class ComplexGatedUnit(nn.Module):
    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        hidden = dim * expand
        self.up = ComplexLinear(dim, hidden * 2, bias=False)
        self.down = ComplexLinear(hidden, dim, bias=False)
        self.norm = ComplexNorm(hidden)
    def forward(self, x):
        h = self.up(x)
        gate, val = h.chunk(2, dim=-2)
        mag = (gate[..., 0]**2 + gate[..., 1]**2).sqrt().unsqueeze(-1)
        activated = val * torch.sigmoid(mag)
        return self.down(self.norm(activated))

class PhaseAssociativeLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, head_dim: int = 64):
        super().__init__()
        inner = n_heads * head_dim
        self.q_proj = ComplexLinear(dim, inner, bias=False)
        self.k_proj = ComplexLinear(dim, inner, bias=False)
        self.v_proj = ComplexLinear(dim, inner, bias=False)
        self.out_proj = ComplexLinear(inner, dim, bias=False)
        self.norm = ComplexNorm(dim)
    def forward(self, x):
        normed = self.norm(x)
        q, k, v = self.q_proj(normed), self.k_proj(normed), self.v_proj(normed)
        return self.out_proj(v)

class ComplexLMHead(nn.Module):
    def __init__(self, dim: int, vocab_size: int = 50257):
        super().__init__()
        self.head_linear = ComplexLinear(dim, dim, bias=False)
        self.head_norm = ComplexNorm(dim)
        self.out_proj = nn.Linear(dim * 2, vocab_size, bias=False)
    def forward(self, x):
        h = self.head_norm(self.head_linear(x))
        flat = h.reshape(*h.shape[:-2], -1)
        return self.out_proj(flat)


class GeneratedModel(nn.Module):
    """Auto-generated model from QLLM Architecture Builder."""

    def __init__(self):
        super().__init__()
        self.embed = ComplexEmbed(vocab_size=50257, dim=256)
        self.cgu_1 = ComplexGatedUnit(dim=256, expand=4)
        self.pam_1 = PhaseAssociativeLayer(dim=256, n_heads=4, head_dim=64)
        self.cgu_2 = ComplexGatedUnit(dim=256, expand=4)
        self.pam_2 = PhaseAssociativeLayer(dim=256, n_heads=4, head_dim=64)
        self.cgu_3 = ComplexGatedUnit(dim=256, expand=4)
        self.pam_3 = PhaseAssociativeLayer(dim=256, n_heads=4, head_dim=64)
        self.head = ComplexLMHead(dim=256, vocab_size=50257)

    def forward(self, input_ids):
        embed_out = self.embed(input_ids)
        cgu_1_out = self.cgu_1(embed_out)
        pam_1_out = self.pam_1(cgu_1_out)
        cgu_2_out = self.cgu_2(pam_1_out)
        pam_2_out = self.pam_2(cgu_2_out)
        cgu_3_out = self.cgu_3(pam_2_out)
        pam_3_out = self.pam_3(cgu_3_out)
        head_out = self.head(pam_3_out)
        return head_out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
