"""
V6 Two-Pass Model: Chunk Encoder + Causal Decoder.

Experimental ablation that processes text in two passes:
1. Non-causal chunk encoder: processes sentence/chunk windows bidirectionally
   to form relational summaries and event candidates.
2. Causal decoder: standard AR prediction with access to encoder summaries
   and episodic memory populated from those summaries.

Uses the same PhaseFieldBackbone for both passes to avoid doubling params.
The encoder pass disables causal masking to let banks see full context.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass

from .config import V6Config
from .init import create_initializer
from .core.complex import (
    ComplexEmbed, ComplexLinear, ComplexNorm, cabs,
)
from .core.ssm import ComplexSSMLayer
from .backbone import PhaseFieldBackbone, BackboneOutput


@dataclass
class TwoPassOutput:
    logits: torch.Tensor
    diversity_loss: Optional[torch.Tensor] = None
    salience: Optional[torch.Tensor] = None
    chunk_summaries: Optional[torch.Tensor] = None


class ChunkEncoder(nn.Module):
    """Bidirectional chunk encoder using forward + backward SSM pass.

    Processes a full chunk of text non-causally by running two SSM
    passes (forward and backward) and combining them. This lets the
    encoder see "Paris is the capital of France" in full before
    deciding what to store.

    Lightweight: reuses bank weights from the backbone and adds only
    a backward SSM layer and a merge projection.
    """

    def __init__(self, config: V6Config, initializer):
        super().__init__()
        self.backward_ssm = ComplexSSMLayer(
            config.dim, config.state_dim,
            dropout=config.dropout,
            initializer=initializer,
        )
        self.merge_proj = ComplexLinear(
            config.dim, config.dim, initializer=initializer,
        )
        self.merge_norm = ComplexNorm(config.dim)
        self.summary_proj = ComplexLinear(
            config.dim, config.dim, initializer=initializer,
        )
        self.summary_norm = ComplexNorm(config.dim)

    def forward(
        self,
        forward_z: torch.Tensor,
        bank_z: torch.Tensor,
    ) -> tuple:
        """
        Args:
            forward_z: [B, L, dim, 2] output from forward SSM pass
            bank_z: [B, L, dim, 2] output from bank+coupler layers

        Returns:
            bidirectional: [B, L, dim, 2] merged forward+backward
            chunk_summary: [B, 1, dim, 2] pooled chunk representation
        """
        backward_input = bank_z.flip(dims=[1])
        backward_h, _ = self.backward_ssm(backward_input)
        backward_z = backward_h.flip(dims=[1])

        merged = self.merge_proj(forward_z + backward_z)
        bidirectional = self.merge_norm(merged)

        mag = cabs(bidirectional)
        weights = mag.mean(dim=-1, keepdim=True)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        summary = (bidirectional * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        chunk_summary = self.summary_norm(self.summary_proj(summary))

        return bidirectional, chunk_summary


class TwoPassLM(nn.Module):
    """Two-pass language model with chunk encoder + causal decoder.

    Pass 1 (Encoder): Run banks + forward SSM (from backbone) + backward SSM
    to get bidirectional chunk representations. Pool into chunk summaries.

    Pass 2 (Decoder): Run a second forward pass through the backbone's SSM
    with chunk summaries injected as additional context. Predict next tokens.

    This is an ablation model: it shares backbone weights but adds a small
    backward SSM and summary projections.
    """

    def __init__(self, config: V6Config):
        super().__init__()
        self.config = config

        initializer = create_initializer(config.init_strategy, config.init_seed)
        config.init_seed = initializer.seed

        self.embed = ComplexEmbed(
            config.vocab_size, config.dim, initializer=initializer,
        )
        self.embed_norm = ComplexNorm(config.dim)

        self.backbone = PhaseFieldBackbone(config, initializer)

        self.chunk_encoder = ChunkEncoder(config, initializer)

        self.decoder_summary_gate = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

        self.lm_head_proj = ComplexLinear(
            config.dim, config.dim, initializer=initializer,
        )
        self.lm_head_norm = ComplexNorm(config.dim)

        self._init_weights()

    def _init_weights(self):
        embed_embeddings = {self.embed.embed_real, self.embed.embed_imag}
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module not in embed_embeddings:
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> TwoPassOutput:
        B, L = input_ids.shape

        z = self.embed(input_ids)
        z = self.embed_norm(z)

        bb = self.backbone(z)

        bidirectional, chunk_summary = self.chunk_encoder(bb.z_out, z)

        gate = self.decoder_summary_gate(cabs(bb.z_out).mean(dim=-1, keepdim=True))
        summary_broadcast = chunk_summary.expand_as(bb.z_out)
        decoder_input = bb.z_out + gate.unsqueeze(-1) * summary_broadcast

        lm = self.lm_head_proj(decoder_input)
        lm = self.lm_head_norm(lm)
        logits = (
            lm[..., 0] @ self.embed.embed_real.weight.T +
            lm[..., 1] @ self.embed.embed_imag.weight.T
        )

        return TwoPassOutput(
            logits=logits,
            diversity_loss=bb.diversity_loss,
            salience=bb.salience,
            chunk_summaries=chunk_summary,
        )

    @property
    def initializer_info(self) -> dict:
        return {
            "init_strategy": self.config.init_strategy,
            "init_seed": self.config.init_seed,
        }

    def count_parameters(self) -> Dict[str, int]:
        bb_counts = self.backbone.count_parameters()
        counts = {
            'embed': sum(p.numel() for p in self.embed.parameters()),
            **bb_counts,
            'chunk_encoder': sum(p.numel() for p in self.chunk_encoder.parameters()),
            'decoder_gate': sum(p.numel() for p in self.decoder_summary_gate.parameters()),
            'lm_head_proj': (
                sum(p.numel() for p in self.lm_head_proj.parameters()) +
                sum(p.numel() for p in self.lm_head_norm.parameters())
            ),
        }
        counts['total'] = sum(counts.values())
        return counts
