# ==========================
# file: quantum_llm_model.py
# ==========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from qllm_utils import causal_mask
import math
from typing import Optional
from torch.utils.checkpoint import checkpoint as activation_checkpoint


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
    """
    Keep the original phase rotator but implemented robustly for variable dims.
    phi is learnable per-dim; tanh ensures [-1,1] then * pi -> [-pi, pi]
    """
    def __init__(self, dim):
        super().__init__()
        self.phase = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: [B, L, D]
        phi = torch.tanh(self.phase) * math.pi
        c, s = torch.cos(phi), torch.sin(phi)
        # apply elementwise per-dim multiplication
        return x * c - x * s

    def coherence_loss(self):
        if self.phase.numel() < 2:
            return torch.tensor(0.0, device=self.phase.device)
        diff = self.phase[1:] - self.phase[:-1]
        return (diff**2).mean() + 0.1 * (self.phase**2).mean()


class InterferenceAttention(nn.Module):
    """
    Implements interference-based attention inspired by the paper:
    - Extract a learned phase per-token (we project embeddings to a 2D sin/cos pair)
    - Compute phase angles via atan2 and interference = cos(delta_phi)
    - Use learnable weight matrix on interference and softmax to compute attn
    This is an experimental replacement for MultiHeadSelfAttention.
    """
    def __init__(self, dim, num_heads, lora_rank=0, lora_alpha=8.0, lora_train_only=False, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # standard q/k/v projections with LoRA support
        self.q = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.k = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.v = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.o = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)

        # small projector to 2D (for sin/cos) per head to compute phase
        self.phase_proj = nn.Linear(dim, 2 * num_heads)  # for each head: (cos, sin)
        # Fix: attention bias projection should be from seq_len to seq_len, not from num_heads
        self.attn_bias_proj = nn.Linear(1, 1)  # Simple scalar projection
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        B, H, L, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * hd)

    def forward(self, x, attn_bias=None):
        # x: [B, L, D]
        q = self._split(self.q(x))
        k = self._split(self.k(x))
        v = self._split(self.v(x))

        # compute classical scores for stability
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # compute per-token phases: project x->(2*num_heads) and reshape [B,L,H,2]
        ph = self.phase_proj(x)  # [B, L, 2*H]
        ph = ph.view(ph.size(0), ph.size(1), self.num_heads, 2)
        # atan2(sin, cos) style angle
        angles = torch.atan2(ph[..., 1], ph[..., 0])  # [B,L,H]

        # compute pairwise delta angles -> interference pattern
        # angles_q = angles.unsqueeze(2)  # [B,L,1,H]
        # angles_k = angles.unsqueeze(1)  # [B,1,L,H]
        # delta = angles_q - angles_k  # broadcasting -> [B,L,L,H]
        # Permute to [B,H,L,L]
        delta = angles.unsqueeze(2) - angles.unsqueeze(1)  # [B,L,L,H]
        delta = delta.permute(0, 3, 1, 2)  # [B,H,L,L]
        interference = torch.cos(delta)  # [B,H,L,L]

        # Fix: properly combine classical scores and interference
        # Take mean across heads and apply simple scaling
        inter_mixed = interference.mean(dim=1, keepdim=True)  # [B,1,L,L]
        inter_mixed = inter_mixed.expand(-1, self.num_heads, -1, -1)  # [B,H,L,L]

        # combine with learnable weight
        scores = scores + 0.1 * inter_mixed  # Small weight to avoid overwhelming classical attention

        if attn_bias is not None:
            scores = scores + attn_bias

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self._merge(out)
        return self.o(out)


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
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, lora_rank=0, lora_alpha=8.0,
                 lora_train_only=False, attention_type: str = "classical"):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = (MultiHeadSelfAttention(dim, num_heads, lora_rank, lora_alpha, lora_train_only, dropout)
                     if attention_type == "classical"
                     else InterferenceAttention(dim, num_heads, lora_rank, lora_alpha, lora_train_only, dropout))
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
                 global_tokens=0, lora_rank=0, lora_alpha=8.0, lora_train_only=False, dropout=0.0,
                 attention_type: str = "classical", use_checkpoint: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_length = seq_length
        self.global_tokens = int(global_tokens)
        self.attention_type = attention_type
        self.use_checkpoint = use_checkpoint

        self.tok_embed = nn.Embedding(vocab_size, dim)
        # pos_embed length = seq_length + global_tokens (we will index accordingly)
        self.pos_embed = nn.Embedding(seq_length + self.global_tokens + 1, dim)
        self.phase = PhaseRotator(dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, 4.0, dropout, lora_rank, lora_alpha, lora_train_only, attention_type)
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
        """
        idx: [B, L]
        returns logits: [B, L, vocab_size] (excludes global tokens positions)
        """
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.tok_embed(idx) + self.pos_embed(pos)
        x = self.phase(x)
        x, g = self.add_globals(x)
        total_len = L + g
        attn_bias = causal_mask(total_len, idx.device)

        # iterate blocks; optionally use activation checkpointing for memory savings
        for blk in self.blocks:
            if self.use_checkpoint:
                # wrap the block call to be checkpointable; checkpoint expects a function that accepts x
                def run_block(x_inner, block=blk, attn_bias=attn_bias):
                    return block(x_inner, attn_bias)
                x = activation_checkpoint(run_block, x, use_reentrant=False)
            else:
                x = blk(x, attn_bias)

        x = self.ln(x)
        logits = self.head(x[:, g:, :])
        return logits

    def phase_coherence_loss(self):
        return self.phase.coherence_loss()
