"""
GPT-2-style Transformer Baseline (~100M params).

A standard decoder-only transformer for apples-to-apples comparison against
PAM models on the same data pipeline. Uses the same GPT-2 tokenizer,
WikiText-103 preprocessing, seq_len=2048, and evaluation loop as v6/train.py.

Architecture:
    - Learned positional embeddings (not RoPE, matching vanilla GPT-2)
    - Pre-norm (LayerNorm before attention and FFN, matching GPT-2 convention)
    - Causal multi-head self-attention via PyTorch scaled_dot_product_attention
    - GELU-activated feed-forward network
    - Tied input/output embeddings

Sizing for ~100M params (to match medium-pam-v3's 100.4M):
    d_model=672, n_layers=12, n_heads=12, d_ff=2688
    This gives ~100.3M params -- within 0.1% of PAM v3's parameter budget.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    max_seq_len: int = 2048
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    tie_weights: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # PyTorch SDPA handles causal masking and flash attention automatically
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only transformer language model (GPT-2-style)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2-style initialization."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)
        # Scale residual projections by 1/sqrt(2*n_layers) as in GPT-2
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out_proj.weight,
                mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
            )
            nn.init.normal_(
                block.ffn.fc2.weight,
                mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, T, vocab_size)."""
        B, T = input_ids.size()
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.drop(self.token_embed(input_ids) + self.pos_embed(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    def count_parameters(self) -> Dict[str, int]:
        embed_params = sum(p.numel() for p in self.token_embed.parameters())
        embed_params += sum(p.numel() for p in self.pos_embed.parameters())
        block_params = sum(
            sum(p.numel() for p in block.parameters())
            for block in self.blocks
        )
        head_params = 0  # tied weights, already counted in embed
        ln_params = sum(p.numel() for p in self.ln_f.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            'embeddings': embed_params,
            'transformer_blocks': block_params,
            'final_ln': ln_params,
            'lm_head': head_params,
            'total': total,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """Simple autoregressive generation (no KV cache)."""
        self.eval()
        ids = input_ids.clone()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            context = ids[:, -self.config.max_seq_len:]
            logits = self.forward(context)[:, -1, :]  # (B, vocab)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(ids[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


def get_transformer_config_100m() -> TransformerConfig:
    """Return a ~100M parameter transformer config.

    d_model=672, n_layers=12, n_heads=12, d_ff=2688  (4x d_model)
    Total: ~100.3M (with tied embeddings) -- matches PAM v3's 100.4M

    Breakdown:
      token_embed: 50257 * 672 = 33.8M
      pos_embed: 2048 * 672 = 1.4M
      per block: 4 * 672^2 (QKV + out) + 2 * 672 * 2688 (FFN) + norms ~ 5.4M
      12 blocks: 65.1M
      Tied head: 0 (shared with token_embed)
      Total: ~100.3M
    """
    return TransformerConfig(
        vocab_size=50257,
        max_seq_len=2048,
        d_model=672,
        n_layers=12,
        n_heads=12,
        d_ff=2688,
        dropout=0.1,
        tie_weights=True,
    )


TRANSFORMER_CONFIGS = {
    '5m': TransformerConfig(
        vocab_size=50257, max_seq_len=2048,
        d_model=88, n_layers=4, n_heads=2, d_ff=352,
        dropout=0.1, tie_weights=True,
    ),
    '10m': TransformerConfig(
        vocab_size=50257, max_seq_len=2048,
        d_model=136, n_layers=13, n_heads=2, d_ff=544,
        dropout=0.1, tie_weights=True,
    ),
    '50m': TransformerConfig(
        vocab_size=50257, max_seq_len=2048,
        d_model=480, n_layers=9, n_heads=7, d_ff=1920,
        dropout=0.1, tie_weights=True,
    ),
    '100m': get_transformer_config_100m(),
}
