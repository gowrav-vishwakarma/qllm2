# ==========================
# file: quantum_llm_model.py
# ==========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from qllm_utils import causal_mask
import math


class LoRALinear(nn.Module):
    """LoRA: y = xW^T + x(BA)^T * (alpha/r), with B:[r,in], A:[out,r]."""
    def __init__(self, in_features, out_features, bias=True, lora_rank=0, lora_alpha=1.0, lora_train_only=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.r = int(lora_rank)
        self.alpha = float(lora_alpha) if self.r > 0 else 1.0
        self.scaling = self.alpha / max(1, self.r)
        self.lora_train_only = lora_train_only
        if self.r > 0:
            # B projects input -> r; A projects r -> out
            self.B = nn.Parameter(torch.zeros(self.r, in_features))
            self.A = nn.Parameter(torch.zeros(out_features, self.r))
            nn.init.zeros_(self.B)
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        out = self.linear(x)
        if self.r > 0:
            update = F.linear(x, self.B)          # [*, r]
            update = F.linear(update, self.A)     # [*, out]
            out = out + self.scaling * update
        return out

    def train(self, mode=True):
        super().train(mode)
        if self.r > 0 and self.lora_train_only:
            for p in self.linear.parameters():
                p.requires_grad_(False)
            if self.A is not None: self.A.requires_grad_(True)
            if self.B is not None: self.B.requires_grad_(True)
        return self


class PhaseRotator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.phase = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        phi = torch.tanh(self.phase) * math.pi
        c, s = torch.cos(phi), torch.sin(phi)
        return x * c - x * s
    def coherence_loss(self):
        diff = self.phase[1:] - self.phase[:-1]
        return (diff**2).mean() + 0.1 * (self.phase**2).mean()


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, lora_rank=0, lora_alpha=8.0, lora_train_only=False, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.k = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.v = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.o = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        B, H, L, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * hd)

    def forward(self, x, attn_bias=None):
        q = self._split(self.q(x))
        k = self._split(self.k(x))
        v = self._split(self.v(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            scores = scores + attn_bias
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self._merge(out)
        return self.o(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, lora_rank=0, lora_alpha=8.0, lora_train_only=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, lora_rank, lora_alpha, lora_train_only, dropout)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x, attn_bias=None):
        x = x + self.attn(self.ln1(x), attn_bias)
        x = x + self.mlp(self.ln2(x))
        return x


class QuantumInspiredLLM(nn.Module):
    def __init__(self, vocab_size=256, dim=384, depth=6, num_heads=8, seq_length=128,
                 global_tokens=0, lora_rank=0, lora_alpha=8.0, lora_train_only=False, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_length = seq_length
        self.global_tokens = int(global_tokens)

        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(seq_length + self.global_tokens, dim)
        self.phase = PhaseRotator(dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, 4.0, dropout, lora_rank, lora_alpha, lora_train_only)
            for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        if self.global_tokens > 0:
            self.global_memory = nn.Parameter(torch.randn(self.global_tokens, dim) * 0.02)
        else:
            self.register_parameter("global_memory", None)

    def add_globals(self, x):
        if self.global_tokens <= 0:
            return x, 0
        B = x.size(0)
        g = self.global_memory.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([g, x], dim=1), self.global_tokens

    def forward(self, idx):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.tok_embed(idx) + self.pos_embed(pos)
        x = self.phase(x)
        x, g = self.add_globals(x)
        total_len = L + g
        attn_bias = causal_mask(total_len, idx.device)
        for blk in self.blocks:
            x = blk(x, attn_bias)
        x = self.ln(x)
        logits = self.head(x[:, g:, :])
        return logits

    def phase_coherence_loss(self):
        return self.phase.coherence_loss()
