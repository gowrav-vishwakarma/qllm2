#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interference Coupler: Combine phase banks via learned interference

Uses Phase2D operations to compute constructive/destructive interference
between different meaning layers (banks).

v2: Added dynamic routing (per-token bank weights) + removed expensive cross_proj.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List

from ..core.interfaces import Coupler
from ..core.registry import register_coupler
from ..core.phase2d import (
    Phase2DLinear, Phase2DLayerNorm,
    phase2d_multiply, phase2d_normalize, phase2d_coherence, phase2d_magnitude
)


@register_coupler('interference', description='Dynamic interference coupling between phase banks')
class InterferenceCoupler(nn.Module):
    """
    Interference Coupler v2: combines banks via DYNAMIC phase interference.
    
    Key features:
    - Per-token dynamic routing (content-dependent bank weights)
    - Learned phase rotations per bank (constructive/destructive interference)
    - GPU-friendly (no trig, all GEMM-based)
    - Removed expensive cross_proj for speed
    
    The key innovation: instead of static bank weights, we compute per-token
    routing weights from the bank outputs themselves. This gives transformer-like
    dynamic interaction without O(n^2) attention.
    """
    
    def __init__(
        self,
        dim: int = 256,
        bank_names: Optional[List[str]] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_dynamic_routing: bool = True,
    ):
        super().__init__()
        self._dim = dim
        self.bank_names = bank_names or ['semantic', 'context']
        self.num_banks = len(self.bank_names)
        self.num_heads = num_heads
        self.use_dynamic_routing = use_dynamic_routing
        
        # Per-bank projection to common interference space
        self.bank_projs = nn.ModuleDict({
            name: Phase2DLinear(dim, dim)
            for name in self.bank_names
        })
        
        # Learned interference phase rotations (complex unit vectors)
        # These create constructive/destructive interference patterns
        self.interference_phases = nn.ParameterDict({
            name: nn.Parameter(torch.randn(dim, 2) * 0.02)
            for name in self.bank_names
        })
        # Initialize close to identity (real=1, imag=0)
        for name in self.bank_names:
            self.interference_phases[name].data[..., 0] = 1.0
            self.interference_phases[name].data[..., 1] = 0.0
        
        # Dynamic routing network: computes per-token bank weights
        # Input: concatenated magnitude features from all banks [batch, seq, num_banks]
        # Output: routing weights [batch, seq, num_banks]
        if use_dynamic_routing:
            self.router = nn.Sequential(
                nn.Linear(self.num_banks, self.num_banks * 4),
                nn.GELU(),
                nn.Linear(self.num_banks * 4, self.num_banks),
            )
            # Initialize router close to uniform
            nn.init.zeros_(self.router[2].weight)
            nn.init.zeros_(self.router[2].bias)
        else:
            self.router = None
            # Fallback: static weights (learnable)
            self.static_weights = nn.Parameter(torch.ones(self.num_banks) / self.num_banks)
        
        # Output fusion (simpler than before - no cross_proj!)
        self.fusion = Phase2DLinear(dim, dim)
        self.norm = Phase2DLayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def name(self) -> str:
        return "interference"
    
    def forward(
        self,
        bank_outputs: Dict[str, torch.Tensor],
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Combine bank outputs via dynamic interference.
        
        Args:
            bank_outputs: {bank_name: [batch, seq, dim, 2]} Phase2D outputs
            context: Optional context
        
        Returns:
            [batch, seq, dim, 2] combined Phase2D output
        """
        # Validate inputs
        available_banks = [name for name in self.bank_names if name in bank_outputs]
        if not available_banks:
            raise ValueError(f"No valid banks found. Expected: {self.bank_names}, got: {list(bank_outputs.keys())}")
        
        # Get batch/seq dimensions from first bank
        first_bank = bank_outputs[available_banks[0]]
        batch_size, seq_len, dim, _ = first_bank.shape
        device = first_bank.device
        
        # 1. Project each bank to interference space + compute magnitude features
        projected = []
        magnitude_features = []
        
        for bank_name in available_banks:
            proj = self.bank_projs[bank_name](bank_outputs[bank_name])
            projected.append(proj)
            
            # Cheap magnitude feature: mean of |z|^2 over dim
            # Shape: [batch, seq]
            mag_sq = (proj[..., 0] ** 2 + proj[..., 1] ** 2).mean(dim=-1)
            magnitude_features.append(mag_sq)
        
        # Stack projected outputs: [num_banks, batch, seq, dim, 2]
        projected = torch.stack(projected, dim=0)
        
        # 2. Compute dynamic routing weights
        if self.use_dynamic_routing and self.router is not None:
            # Stack magnitude features: [batch, seq, num_banks]
            mag_features = torch.stack(magnitude_features, dim=-1)
            
            # Router produces logits, then softmax for weights
            routing_logits = self.router(mag_features)  # [batch, seq, num_banks]
            routing_weights = F.softmax(routing_logits, dim=-1)  # [batch, seq, num_banks]
        else:
            # Static weights (broadcast to all positions)
            routing_weights = F.softmax(self.static_weights, dim=0)  # [num_banks]
            routing_weights = routing_weights.view(1, 1, self.num_banks).expand(batch_size, seq_len, -1)
        
        # 3. Apply interference phase rotations (complex multiply)
        interfered = []
        for i, bank_name in enumerate(available_banks):
            phase = self.interference_phases[bank_name]  # [dim, 2]
            # Normalize to unit magnitude for pure rotation
            phase_normalized = phase2d_normalize(phase)  # [dim, 2]
            phase_normalized = phase_normalized.unsqueeze(0).unsqueeze(0)  # [1, 1, dim, 2]
            
            # Apply rotation: bank * e^(i*theta)
            bank_rotated = phase2d_multiply(projected[i], phase_normalized)
            interfered.append(bank_rotated)
        
        # Stack: [num_banks, batch, seq, dim, 2]
        interfered = torch.stack(interfered, dim=0)
        
        # 4. Weighted sum with dynamic routing
        # routing_weights: [batch, seq, num_banks] -> [num_banks, batch, seq, 1, 1]
        weights = routing_weights.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)
        
        # Weighted interference: sum over banks
        combined = (interfered * weights).sum(dim=0)  # [batch, seq, dim, 2]
        
        # 5. Output fusion
        output = self.norm(combined)
        output = self.fusion(output)
        
        # Dropout (applied to real part, preserving phase structure)
        if self.training:
            output = self.dropout(output[..., 0]).unsqueeze(-1).expand_as(output) * output + output
        
        return output
    
    def compute_coupling_loss(
        self,
        bank_outputs: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Compute cross-bank diversity loss.
        
        Changed from original: now encourages DIVERSITY between banks,
        not just alignment. Banks should capture different aspects of the signal.
        
        Loss = -variance(coherences) + mean(|coherence|) 
        This encourages some bank pairs to align, others to oppose.
        """
        available_banks = [name for name in self.bank_names if name in bank_outputs]
        if len(available_banks) < 2:
            return None
        
        # Subsample for memory efficiency
        max_tokens_for_loss = 32  # Reduced from 64 for speed
        
        coherences = []
        
        for i, name_i in enumerate(available_banks):
            for name_j in available_banks[i+1:]:
                bank_i = bank_outputs[name_i]
                bank_j = bank_outputs[name_j]
                
                # Reduce sequence dimension
                if bank_i.dim() == 4:
                    seq_len = bank_i.shape[1]
                    if seq_len > max_tokens_for_loss:
                        # Faster: just take first K instead of linspace
                        bank_i = bank_i[:, :max_tokens_for_loss]
                        bank_j = bank_j[:, :max_tokens_for_loss]
                    bank_i = bank_i.mean(dim=1)
                    bank_j = bank_j.mean(dim=1)
                
                coherence = phase2d_coherence(bank_i, bank_j)
                coherences.append(coherence.mean())
        
        if len(coherences) == 0:
            return None
        
        coherences = torch.stack(coherences)
        
        # New loss: encourage diversity (variance) while keeping magnitudes moderate
        # High variance = some banks align, others oppose = good!
        # But don't let coherence magnitudes explode
        if len(coherences) > 1:
            diversity_loss = -coherences.var() + 0.1 * (coherences.abs().mean() - 0.5).abs()
        else:
            # Only one pair: just encourage moderate coherence (not too aligned, not too opposed)
            diversity_loss = 0.1 * (coherences.abs().mean() - 0.5).abs()
        
        return diversity_loss
