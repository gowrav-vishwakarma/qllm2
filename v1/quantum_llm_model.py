# ==========================
# file: quantum_llm_model.py
# ==========================
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from qllm_utils import causal_mask


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
            update = F.linear(x, self.B)      # [*, r]
            update = F.linear(update, self.A) # [*, out]
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
    """Learnable per-dim phase; mild smoothness regularizer."""
    def __init__(self, dim, init_scale: float = 1e-2):
        super().__init__()
        self.phase = nn.Parameter(torch.randn(dim) * float(init_scale))

    def forward(self, x):
        phi = torch.tanh(self.phase) * math.pi
        c, s = torch.cos(phi), torch.sin(phi)
        return x * c - x * s

    def coherence_loss(self):
        if self.phase.numel() < 2:
            return torch.tensor(0.0, device=self.phase.device)
        diff = self.phase[1:] - self.phase[:-1]
        return (diff**2).mean() + 0.1 * (self.phase**2).mean()


class InterferenceAttention(nn.Module):
    """
    Fast interference-based attention.

    Key speedups vs previous version:
    - Avoids building angle pairwise differences (no LxL trig); uses Gram form:
        cos(Δϕ) = cosϕ_i cosϕ_j + sinϕ_i sinϕ_j
      implemented as P @ P^T with P = [cosϕ, sinϕ] (2-dim features per token per head).
    - Uses PyTorch scaled_dot_product_attention (Flash/Math/Memory-efficient paths),
      adding the interference term as an additive attention bias.

    Knobs:
      - interference_beta: global scale for the interference bias.
      - inter_heads_fraction: fraction of heads that participate (others get 0 bias).
      - per-head learned gate (sigmoid) to let the model tune contribution.
    """
    def __init__(self, dim, num_heads, lora_rank=0, lora_alpha=8.0, lora_train_only=False, dropout=0.0,
                 interference_beta: float = 0.08, inter_heads_fraction: float = 1.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = float(dropout)

        # q/k/v with LoRA
        self.q = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.k = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.v = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.o = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)

        # 2-dim phase features per head (cos,sin); we normalize to unit length
        self.phase_proj = nn.Linear(dim, 2 * num_heads)
        # learned per-head gate in [0,1] via sigmoid; init ~0.1
        init_logit = -2.197224577  # sigmoid ~ 0.10
        self.gamma = nn.Parameter(torch.full((num_heads,), init_logit))
        # global scale
        self.interference_beta = float(interference_beta)

        # only a fraction of heads are allowed to use interference (others get gamma -> -inf effectively)
        h_active = max(1, int(round(inter_heads_fraction * num_heads)))
        mask = torch.zeros(num_heads)
        mask[:h_active] = 1.0
        self.register_buffer("active_heads_mask", mask)

    def _split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,hd]

    def _merge(self, x):
        B, H, L, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * hd)

    def forward(self, x, attn_bias=None):
        # q/k/v
        q = self._split(self.q(x))  # [B,H,L,hd]
        k = self._split(self.k(x))
        v = self._split(self.v(x))

        B, H, L, _ = q.shape

        # Build interference bias as Gram matrix of 2D features per head
        # phase features: [B,L,2H] -> [B,H,L,2]
        ph = self.phase_proj(x).view(B, L, self.num_heads, 2).permute(0, 2, 1, 3)  # [B,H,L,2]
        # normalize to unit circle to stabilize training
        ph = F.normalize(ph, p=2, dim=-1, eps=1e-6)
        c = ph[..., 0]  # [B,H,L]
        s = ph[..., 1]  # [B,H,L]

        # Gram: (c c^T + s s^T) -> [B,H,L,L]
        # do as outer products via broadcasting (fast on GPU, no trig per pair)
        inter_bias = (c.unsqueeze(-1) * c.unsqueeze(-2)) + (s.unsqueeze(-1) * s.unsqueeze(-2))

        # per-head gate in [0,1], apply fraction mask, and global beta
        head_gate = torch.sigmoid(self.gamma) * self.active_heads_mask  # [H]
        inter_bias = inter_bias * head_gate.view(1, H, 1, 1) * self.interference_beta

        # compose final additive mask: causal + interference
        # attn_bias is causal mask shaped [1,1,L,L] with -inf above diagonal; upcast to q dtype
        if attn_bias is None:
            mask = inter_bias
        else:
            mask = attn_bias.to(q.dtype) + inter_bias

        # scaled_dot_product_attention uses efficient kernels (Flash/Math/Memory) automatically
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=mask,
                                             dropout_p=self.dropout_p if self.training else 0.0,
                                             is_causal=False)
        out = self._merge(out)
        return self.o(out)


class MultiHeadSelfAttention(nn.Module):
    """SDPA-based MHA (faster than manual matmuls)."""
    def __init__(self, dim, num_heads, lora_rank=0, lora_alpha=8.0, lora_train_only=False, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = float(dropout)
        self.q = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.k = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.v = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)
        self.o = LoRALinear(dim, dim, True, lora_rank, lora_alpha, lora_train_only)

    def _split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,hd]

    def _merge(self, x):
        B, H, L, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * hd)

    def forward(self, x, attn_bias=None):
        q = self._split(self.q(x))
        k = self._split(self.k(x))
        v = self._split(self.v(x))
        mask = None if attn_bias is None else attn_bias.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=mask,
                                             dropout_p=self.dropout_p if self.training else 0.0,
                                             is_causal=False)
        out = self._merge(out)
        return self.o(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, lora_rank=0, lora_alpha=8.0,
                 lora_train_only=False, attention_type: str = "classical",
                 interference_beta: float = 0.08, inter_heads_fraction: float = 1.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        if attention_type == "classical":
            self.attn = MultiHeadSelfAttention(dim, num_heads, lora_rank, lora_alpha, lora_train_only, dropout)
        else:
            self.attn = InterferenceAttention(dim, num_heads, lora_rank, lora_alpha, lora_train_only, dropout,
                                              interference_beta=interference_beta,
                                              inter_heads_fraction=inter_heads_fraction)
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
                 attention_type: str = "classical", use_checkpoint: bool = False,
                 interference_beta: float = 0.08, inter_heads_fraction: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_length = seq_length
        self.global_tokens = int(global_tokens)
        self.attention_type = attention_type
        self.use_checkpoint = use_checkpoint

        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(seq_length + self.global_tokens + 1, dim)
        self.phase = PhaseRotator(dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, 4.0, dropout, lora_rank, lora_alpha, lora_train_only,
                             attention_type, interference_beta, inter_heads_fraction)
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
        attn_bias = causal_mask(total_len, idx.device)  # [1,1,T,T] with -inf above diag

        for blk in self.blocks:
            if self.use_checkpoint:
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
