"""Gated DeltaNet baseline on the v13_sempty training path (closest prior work for PAM's delta notebook).

Reference: Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv 2412.06464).
Implementation: the published flash-linear-attention ``fla`` 0.3.2 (``fla.models.GatedDeltaNetModel``:
GatedDeltaNet mixer with short conv on q / k / v, output gate, SwiGLU MLP, RMSNorm; chunked Triton kernels).
Checked in fla/layers/gated_deltanet.py: NO rotary / phase on q or k (position enters only through the short conv
and the gated decay), which is the point of the comparison with PAM's Chrono clock.

Same interface as ``baseline_mamba2.Mamba2LM`` (``_hidden_to_lm`` / ``ce_from_lm`` with the SAME fused CE and
answer-weight loss_mask, ``forward``, ``generate``, ``count_parameters``, dataclass ``config``), so data, loss,
schedule, checkpoints and every probe are shared; the only variable is the sequence mixer. Raw torch by design
(third-party reference) -> listed in ``check_torch_layout.SKIP_FILES``.

Size: d 768, 12 layers, 6 heads x head_dim 64 (expand_v 2), SwiGLU 1280, tied embedding padded to 50,264 rows
-> ~102.5M, matched to ctl PAM M2 (102.88M) within 1 %.

NOT a pip dep. Install (cp311 venv, keep torch / triton / transformers):
  ~/.local/bin/uv pip install --python ~/.venvs/qllm311/bin/python --no-deps flash-linear-attention==0.3.2 fla-core==0.3.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from v13_sempty.baseline_mamba2 import Mamba2LM


@dataclass
class GDNConfig:
    arch: str = 'gdn'
    vocab_size: int = 50261
    d_model: int = 768
    n_layer: int = 12
    num_heads: int = 6
    head_dim: int = 64
    expand_v: float = 2.0
    intermediate_size: int = 1280
    conv_size: int = 4
    pad_vocab: int = 50264


class GDNLM(nn.Module):
    """fla GatedDeltaNetModel backbone + tied head (no separate lm_head weight)."""

    def __init__(self, cfg: GDNConfig):
        super().__init__()
        from fla.models import GatedDeltaNetConfig, GatedDeltaNetModel
        self.config = cfg
        self.net = GatedDeltaNetModel(GatedDeltaNetConfig(
            hidden_size=cfg.d_model, num_hidden_layers=cfg.n_layer, num_heads=cfg.num_heads,
            head_dim=cfg.head_dim, expand_v=cfg.expand_v, intermediate_size=cfg.intermediate_size,
            conv_size=cfg.conv_size, vocab_size=cfg.pad_vocab, tie_word_embeddings=True,
            fuse_cross_entropy=False, use_cache=False))

    @property
    def blocks(self) -> nn.ModuleList:
        return self.net.layers

    def _head_weight(self) -> torch.Tensor:
        return self.net.embeddings.weight[: self.config.vocab_size]

    def _hidden_to_lm(self, input_ids, step_offset: int = 0):
        hidden = self.net(input_ids=input_ids, use_cache=False).last_hidden_state   # after the final norm
        return hidden, torch.zeros((), device=input_ids.device)

    ce_from_lm = Mamba2LM.ce_from_lm
    generate = Mamba2LM.generate

    def forward(self, input_ids, states=None, step_offset: int = 0, labels=None):
        hidden, aux = self._hidden_to_lm(input_ids)
        return F.linear(hidden, self._head_weight()), None, aux

    def count_parameters(self) -> Dict[str, int]:
        embed_p = self.net.embeddings.weight.numel()
        block_p = sum(p.numel() for p in self.blocks.parameters())
        norm_p = sum(p.numel() for p in self.net.norm.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {'embedding (tied)': embed_p, 'blocks': block_p, 'norms': norm_p, 'total': total}
