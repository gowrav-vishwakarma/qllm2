"""
Phase Interference Coupler for V6.

Revives V4's InterferenceCoupler with V5-safe ops. Combines bank outputs
via learned phase rotations and dynamic routing weights. Each bank's output
is rotated by a learned complex phase before being mixed.

No phase-breaking ops: routing uses complex magnitude features, phase
rotations use complex multiplication with learned unit-complex weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..init import InitStrategy

from .complex import (
    ComplexLinear, ComplexNorm,
    cmul, cnormalize, cabs, cabs2, creal_dot, to_real,
)


class PhaseInterferenceCoupler(nn.Module):
    """
    Combines outputs from named banks via phase interference.

    Each source gets a learned phase rotation (unit complex per dim) and
    a dynamic routing weight (content-dependent, based on magnitudes).
    The mixed output is projected and normalized.

    output = norm(proj(sum_i(route_i * rotate(source_i, phase_i))))
    """

    def __init__(
        self,
        dim: int,
        num_sources: int = 2,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_sources = num_sources

        # Learned phase rotations: one unit-complex vector per source
        if initializer is not None:
            self.phase_rotations = nn.ParameterList([
                nn.Parameter(initializer.init_phase_rotation(dim))
                for _ in range(num_sources)
            ])
        else:
            rots = []
            for _ in range(num_sources):
                rot = torch.zeros(dim, 2)
                rot[:, 0] = 1.0
                rot[:, 1] = torch.randn(dim) * 0.01
                rots.append(nn.Parameter(rot))
            self.phase_rotations = nn.ParameterList(rots)

        # Per-source projection (complex)
        self.source_projs = nn.ModuleList([
            ComplexLinear(dim, dim, bias=False, initializer=initializer)
            for _ in range(num_sources)
        ])

        # Router: uses magnitude^2 features from each bank as input
        router_in = dim * num_sources
        self.router = nn.Sequential(
            nn.Linear(router_in, dim),
            nn.GELU(),
            nn.Linear(dim, num_sources),
        )

        # Output projection and norm
        self.out_proj = ComplexLinear(dim, dim, initializer=initializer)
        self.norm = ComplexNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *sources: torch.Tensor) -> torch.Tensor:
        """
        sources: tuple of [B, L, dim, 2] tensors (one per bank).
        Returns: [B, L, dim, 2] combined output.
        """
        assert len(sources) == self.num_sources

        # Compute routing weights from magnitude features
        mag_features = [cabs(s) for s in sources]  # list of [B, L, dim]
        router_in = torch.cat(mag_features, dim=-1)  # [B, L, dim*num_sources]
        route_weights = F.softmax(self.router(router_in), dim=-1)  # [B, L, num_sources]

        # Apply phase rotation and routing to each source
        combined = torch.zeros_like(sources[0])
        for i, (src, proj, phase_rot) in enumerate(
            zip(sources, self.source_projs, self.phase_rotations)
        ):
            # Project source
            projected = proj(src)  # [B, L, dim, 2]

            # Rotate by learned phase (complex multiply with unit-complex)
            unit_rot = cnormalize(phase_rot)  # [dim, 2] -> unit complex
            rotated = cmul(unit_rot.unsqueeze(0).unsqueeze(0), projected)

            # Weight by routing
            weight = route_weights[..., i].unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
            combined = combined + rotated * weight

        # Output projection
        out = self.out_proj(combined)
        out = self.norm(out)

        if self.training:
            mask = self.dropout(torch.ones(out.shape[:-1], device=out.device))
            out = out * mask.unsqueeze(-1)

        return out

    def compute_diversity_loss(self) -> torch.Tensor:
        """
        Encourage phase rotations to be diverse (different angles per bank).
        """
        if self.num_sources < 2:
            return torch.tensor(0.0)

        losses = []
        for i in range(self.num_sources):
            for j in range(i + 1, self.num_sources):
                ri = cnormalize(self.phase_rotations[i])
                rj = cnormalize(self.phase_rotations[j])
                sim = creal_dot(ri.unsqueeze(0), rj.unsqueeze(0)).abs()
                losses.append(sim.mean())

        return torch.stack(losses).mean() if losses else torch.tensor(0.0)
