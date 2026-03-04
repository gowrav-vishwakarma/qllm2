"""
Multi-bank system with algebraic fusion.

Each bank is a ComplexGatedUnit that learns to specialize on different
aspects of the input. Banks occupy different phase subspaces and their
outputs combine via genuine complex interference.

The router determines PER-TOKEN which bank contributions matter most,
using complex routing weights that both SCALE and ROTATE each bank's output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..init import InitStrategy

from .complex import (
    ComplexLinear, ComplexNorm, ComplexGatedUnit,
    cmul, cnormalize, cabs, creal_dot, to_real,
)


class AlgebraicBank(nn.Module):
    """
    One processing bank: a ComplexGatedUnit with pre-norm.

    Banks learn to specialize through diversity loss and training dynamics.
    No hand-designed roles (semantic, context, etc.) -- let the algebra learn.
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.cgu = ComplexGatedUnit(dim, expand, initializer=initializer)
        self.norm = ComplexNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, L, dim, 2] -> [B, L, dim, 2]"""
        out = self.cgu(self.norm(z))
        if self.training:
            mask = self.dropout(torch.ones(out.shape[:-1], device=out.device))
            out = out * mask.unsqueeze(-1)
        return out


class ComplexRouter(nn.Module):
    """
    Content-dependent routing with complex weights.

    Routes based on input content: magnitude determines HOW MUCH of each
    bank to use, phase determines WHAT ROTATION to apply.

    Uses the magnitude features from all banks for routing decisions.
    """

    def __init__(self, dim: int, num_banks: int):
        super().__init__()
        self.num_banks = num_banks
        # Router MLP: input magnitude features -> bank logits
        self.router = nn.Sequential(
            nn.Linear(num_banks, num_banks * 4),
            nn.GELU(),
            nn.Linear(num_banks * 4, num_banks),
        )
        # Initialize near-uniform
        nn.init.zeros_(self.router[2].weight)
        nn.init.zeros_(self.router[2].bias)

    def forward(self, bank_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute per-token routing weights from bank output magnitudes.

        Args:
            bank_outputs: list of [B, L, dim, 2] tensors

        Returns:
            weights: [B, L, num_banks] routing probabilities
        """
        # Cheap magnitude features: mean |z|^2 per bank
        mag_features = []
        for bo in bank_outputs:
            mag = (bo[..., 0].square() + bo[..., 1].square()).mean(dim=-1)  # [B, L]
            mag_features.append(mag)

        mag_feat = torch.stack(mag_features, dim=-1)  # [B, L, num_banks]
        logits = self.router(mag_feat)
        return F.softmax(logits, dim=-1)


class AlgebraicFusion(nn.Module):
    """
    Combine bank outputs via learned phase rotations and dynamic routing.

    Each bank has a learned phase rotation (unit complex per dim).
    The router provides per-token weights. The combination is:

        output = sum_i( weight_i * rotate(bank_i, phase_i) )

    This IS genuine algebraic interference: bank outputs at different phases
    constructively/destructively combine based on content-dependent routing.
    """

    def __init__(
        self,
        dim: int,
        num_banks: int,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_banks = num_banks

        if initializer is not None:
            phase_rot = [nn.Parameter(initializer.init_phase_rotation(dim)) for _ in range(num_banks)]
        else:
            phase_rot = [nn.Parameter(self._init_near_identity(dim)) for _ in range(num_banks)]
        self.phase_rotations = nn.ParameterList(phase_rot)

        self.bank_projs = nn.ModuleList([
            ComplexLinear(dim, dim, bias=False, initializer=initializer)
            for _ in range(num_banks)
        ])

        self.router = ComplexRouter(dim, num_banks)
        self.output_proj = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.norm = ComplexNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _init_near_identity(dim: int) -> torch.Tensor:
        """Initialize phase rotation near identity (real=1, imag≈0)."""
        rot = torch.zeros(dim, 2)
        rot[:, 0] = 1.0
        rot[:, 1] = torch.randn(dim) * 0.01
        return rot

    def forward(
        self,
        bank_outputs: List[torch.Tensor],
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combine bank outputs via interference.

        Args:
            bank_outputs: list of [B, L, dim, 2]
            x: original input [B, L, dim, 2] (unused, for API compatibility)

        Returns:
            [B, L, dim, 2]
        """
        # Project each bank and apply phase rotation
        rotated = []
        for i, (bo, proj, phase) in enumerate(
            zip(bank_outputs, self.bank_projs, self.phase_rotations)
        ):
            projected = proj(bo)
            # Normalize phase to unit magnitude for pure rotation
            unit_phase = cnormalize(phase)  # [dim, 2]
            # Apply rotation: broadcast over [B, L]
            rot_out = cmul(projected, unit_phase.unsqueeze(0).unsqueeze(0))
            rotated.append(rot_out)

        # Dynamic routing weights
        weights = self.router(bank_outputs)  # [B, L, num_banks]

        # Weighted interference
        combined = torch.zeros_like(rotated[0])
        for i, rot_out in enumerate(rotated):
            w = weights[..., i].unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
            combined = combined + w * rot_out

        # Output projection
        out = self.norm(combined)
        out = self.output_proj(out)

        return out

    def compute_diversity_loss(
        self,
        bank_outputs: List[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Encourage banks to specialize by minimizing inter-bank similarity.

        Uses complex cosine similarity: Re(a*conj(b)) / (|a|*|b|).
        We want this to be LOW between different banks (diverse outputs).
        """
        if len(bank_outputs) < 2:
            return None

        similarities = []
        for i in range(len(bank_outputs)):
            for j in range(i + 1, len(bank_outputs)):
                a = bank_outputs[i]
                b = bank_outputs[j]

                # Subsample for efficiency
                if a.shape[1] > 32:
                    a = a[:, :32]
                    b = b[:, :32]

                # Complex cosine similarity per position, averaged
                # Re(a*conj(b)) / (|a|*|b|)
                dot = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(dim=-1)
                mag_a = cabs(a).sum(dim=-1)
                mag_b = cabs(b).sum(dim=-1)
                cos_sim = dot / (mag_a * mag_b + 1e-8)
                similarities.append(cos_sim.abs().mean())

        return torch.stack(similarities).mean()


class MultiBank(nn.Module):
    """
    Top-level multi-bank module: N banks + fusion.

    Wraps AlgebraicBanks and AlgebraicFusion into a single module
    that takes input and returns fused output + diversity loss.
    """

    def __init__(
        self,
        dim: int,
        num_banks: int = 2,
        expand: int = 2,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.num_banks = num_banks

        self.banks = nn.ModuleList([
            AlgebraicBank(dim, expand, dropout, initializer=initializer)
            for _ in range(num_banks)
        ])

        self.fusion = AlgebraicFusion(dim, num_banks, dropout, initializer=initializer)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, L, dim, 2] -> [B, L, dim, 2]"""
        bank_outputs = [bank(z) for bank in self.banks]
        return self.fusion(bank_outputs, z)

    def compute_diversity_loss(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Run banks and compute diversity loss (for training)."""
        bank_outputs = [bank(z) for bank in self.banks]
        return self.fusion.compute_diversity_loss(bank_outputs)
