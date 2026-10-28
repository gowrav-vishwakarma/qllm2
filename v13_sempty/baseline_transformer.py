"""Transformer++ baseline on the v13_sempty training path (equal-data reference).

The standard modern attention baseline of the linear-RNN / SSM literature
("Transformer++": Llama-style pre-norm decoder -- RMSNorm, RoPE, SwiGLU, no
biases, tied embeddings, causal SDPA / flash attention) behind the interface
``train.Trainer`` and the probes use from ``model.LM`` (same as
``baseline_mamba2.py``):

  ``_hidden_to_lm`` / ``ce_from_lm``  the SAME fused chunked CE as PAM
                                      (answer-weight ``loss_mask`` included)
  ``forward`` -> (logits, None, aux)  probes / PG-19 eval
  ``generate``, ``count_parameters``, ``config`` (dataclass, saved in ckpts)

so data, loss, LR schedule, checkpoints, resume and every eval are shared
with PAM; the only variable is the sequence mixer. NOT O(1): attention keeps
every past token (O(T) KV cache, O(T^2) prefill) -- it is the quality bar,
not a competitor on state size. RoPE tables grow on demand, so the recall
probes run past the 2048 training window (extrapolation, as for PAM/Mamba-2).
Raw torch by design (a reference model) -> ``check_torch_layout.SKIP_FILES``.

Default size, param-matched to the PAM reference (102.1M): d_model 672, 12
layers, 12 heads (head_dim 56), SwiGLU hidden 1920 -> 101.9M with the
50,261-token chat vocab. Same width/depth/heads as the 100.3M GPT-2-style
transformer of the WikiText fair run (``v6/transformer_baseline.py``, 22.69).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    arch: str = 'transformer'
    vocab_size: int = 50261
    d_model: int = 672
    n_layer: int = 12
    n_head: int = 12
    d_ff: int = 1920
    rope_base: float = 10000.0
    norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


def _rotate(x, cos, sin):
    """Rotate-half RoPE on [B, H, T, hd] with cos/sin [T, hd/2]."""
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos[None, None], sin[None, None]
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class TransformerLM(nn.Module):
    """Transformer++ decoder, tied head."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0 and (cfg.d_model // cfg.n_head) % 2 == 0
        self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        hd = cfg.d_model // cfg.n_head
        self.register_buffer(
            'inv_freq', 1.0 / (cfg.rope_base ** (torch.arange(0, hd, 2).float() / hd)),
            persistent=False)
        self._init_weights()

    def _init_weights(self):
        """GPT-2 style: N(0, 0.02); residual-out projections / sqrt(2L)."""
        std_out = 0.02 / math.sqrt(2 * self.config.n_layer)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=std_out if name.endswith(('o_proj.weight', 'down.weight'))
                                else 0.02)

    @property
    def blocks(self) -> nn.ModuleList:
        """The decoder layers (Trainer: per-block grad norms)."""
        return self.layers

    def _head_weight(self) -> torch.Tensor:
        return self.embed.weight

    def _rope(self, T: int, device):
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))     # [T, hd/2] fp32
        return freqs.cos(), freqs.sin()

    def _hidden_to_lm(self, input_ids, step_offset: int = 0):
        cos, sin = self._rope(input_ids.shape[1], input_ids.device)
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm_f(x), torch.zeros((), device=input_ids.device)

    def ce_from_lm(self, lm, labels, loss_mask=None, ignore_index=-100,
                   chunk: int = 4096, return_nll: bool = False, gemm_dtype=None):
        """The PAM loss: ``v13_sempty.fused_ce`` on (hidden, tied embedding)."""
        from v13_sempty.fused_ce import fused_linear_cross_entropy
        b, t, d = lm.shape
        out = fused_linear_cross_entropy(
            lm.reshape(b * t, d), self._head_weight(), labels.reshape(-1),
            mask=(loss_mask.reshape(-1) if loss_mask is not None else None),
            chunk=chunk, ignore_index=ignore_index, return_nll=return_nll,
            gemm_dtype=gemm_dtype,
        )
        if return_nll:
            return out, getattr(out, '_nll', None).reshape(b, t)
        return out

    def forward(self, input_ids, states=None, step_offset: int = 0, labels=None):
        hidden, aux = self._hidden_to_lm(input_ids)
        return F.linear(hidden, self._head_weight()), None, aux

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0,
                 top_k=50, top_p=0.0, repetition_penalty=1.0, eos_token_id=None):
        """Sampling for the in-training text samples (full re-forward per
        token, no KV cache: only ever ~80 tokens)."""
        self.eval()
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)[0][:, -1].float()
            if repetition_penalty != 1.0:
                score = torch.gather(logits, 1, generated)
                score = torch.where(score > 0, score / repetition_penalty,
                                    score * repetition_penalty)
                logits.scatter_(1, generated, score)
            if temperature <= 0.0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = logits.topk(min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = float('-inf')
                if top_p > 0:
                    sl, si = logits.sort(descending=True)
                    p = sl.softmax(dim=-1)
                    sl[p.cumsum(dim=-1) - p >= top_p] = float('-inf')
                    logits = sl.scatter(1, si, sl)
                nxt = torch.multinomial(logits.softmax(dim=-1), 1)
            generated = torch.cat([generated, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
        return generated

    def count_parameters(self) -> Dict[str, int]:
        embed_p = self.embed.weight.numel()
        block_p = sum(p.numel() for p in self.layers.parameters())
        norm_p = sum(p.numel() for p in self.norm_f.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {'embedding (tied)': embed_p, 'blocks': block_p,
                'norms': norm_p, 'total': total}
