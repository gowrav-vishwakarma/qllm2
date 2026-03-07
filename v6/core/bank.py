"""
Named banks for V6: SemanticBank and ContextBank.

Revives V4's named-bank concept but built with V5-safe complex ops only.
Each bank is a ComplexGatedUnit with pre-norm -- no phase-breaking ops anywhere.
Banks specialize through training dynamics and diversity loss.
"""

import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..init import InitStrategy

from .complex import (
    ComplexLinear, ComplexNorm, ComplexGatedUnit,
    cmul, cnormalize, cabs, creal_dot, to_real,
)


class SemanticBank(nn.Module):
    """
    Processes semantic content: word meanings, entity relationships, concepts.
    Internally a ComplexGatedUnit with pre-norm -- pure phase-safe ops.
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.norm = ComplexNorm(dim)
        self.cgu = ComplexGatedUnit(dim, expand, initializer=initializer)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, L, dim, 2] -> [B, L, dim, 2]"""
        out = self.cgu(self.norm(z))
        if self.training:
            mask = self.dropout(torch.ones(out.shape[:-1], device=out.device))
            out = out * mask.unsqueeze(-1)
        return out


class ContextBank(nn.Module):
    """
    Processes contextual/positional patterns: syntax, structure, discourse.
    Same architecture as SemanticBank, specialization through training.
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.norm = ComplexNorm(dim)
        self.cgu = ComplexGatedUnit(dim, expand, initializer=initializer)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.cgu(self.norm(z))
        if self.training:
            mask = self.dropout(torch.ones(out.shape[:-1], device=out.device))
            out = out * mask.unsqueeze(-1)
        return out


class NamedBankPair(nn.Module):
    """
    A pair of named banks (semantic + context) for one layer.

    Provides diversity loss to encourage specialization between the two banks.
    The coupler (separate module) combines their outputs.
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.semantic = SemanticBank(dim, expand, dropout, initializer)
        self.context = ContextBank(dim, expand, dropout, initializer)

    def forward(self, z: torch.Tensor) -> tuple:
        """Returns (semantic_out, context_out) each [B, L, dim, 2]."""
        return self.semantic(z), self.context(z)

    def compute_diversity_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encourage banks to specialize by penalizing similarity between their outputs.
        Uses complex cosine similarity: Re(a * conj(b)) / (|a| * |b|).
        """
        sem_out = self.semantic(z)
        ctx_out = self.context(z)

        sem_flat = sem_out.reshape(-1, sem_out.shape[-2], 2)
        ctx_flat = ctx_out.reshape(-1, ctx_out.shape[-2], 2)

        dot = creal_dot(sem_flat, ctx_flat)
        sem_mag = torch.sqrt(cabs(sem_flat).square().sum(dim=-1) + 1e-8)
        ctx_mag = torch.sqrt(cabs(ctx_flat).square().sum(dim=-1) + 1e-8)

        cosine_sim = dot / (sem_mag * ctx_mag + 1e-8)
        return cosine_sim.abs().mean()
