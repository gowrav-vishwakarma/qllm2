#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interference Coupler: Combine phase banks via learned interference

Uses Phase2D operations to compute constructive/destructive interference
between different meaning layers (banks).
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


@register_coupler('interference', description='Learned interference coupling between phase banks')
class InterferenceCoupler(nn.Module):
    """
    Interference Coupler: combines banks via phase interference.
    
    Key features:
    - Learned interference weights per bank
    - Cross-bank phase alignment
    - Constructive/destructive interference via complex ops
    - GPU-friendly (no trig)
    """
    
    def __init__(
        self,
        dim: int = 256,
        bank_names: Optional[List[str]] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._dim = dim
        self.bank_names = bank_names or ['semantic', 'context']
        self.num_banks = len(self.bank_names)
        self.num_heads = num_heads
        
        # Per-bank projection to common interference space
        self.bank_projs = nn.ModuleDict({
            name: Phase2DLinear(dim, dim)
            for name in self.bank_names
        })
        
        # Learned interference weights (complex-valued)
        # These determine constructive vs destructive interference
        self.interference_weights = nn.ParameterDict({
            name: nn.Parameter(torch.randn(dim, 2) * 0.02)
            for name in self.bank_names
        })
        # Initialize with positive real (constructive by default)
        for name in self.bank_names:
            self.interference_weights[name].data[..., 0] = 1.0 / self.num_banks
            self.interference_weights[name].data[..., 1] = 0.0
        
        # Cross-bank interaction (attention-like but with phase)
        self.cross_proj = Phase2DLinear(dim * self.num_banks, dim)
        
        # Output fusion
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
        Combine bank outputs via interference.
        
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
        
        # 1. Project each bank to interference space
        projected = {}
        for bank_name in available_banks:
            proj = self.bank_projs[bank_name](bank_outputs[bank_name])
            projected[bank_name] = proj
        
        # 2. Apply interference weights (complex multiplication)
        interfered = []
        for bank_name in available_banks:
            weight = self.interference_weights[bank_name]  # [dim, 2]
            weight = weight.unsqueeze(0).unsqueeze(0)  # [1, 1, dim, 2]
            
            # Interference = bank_output * weight (complex multiply)
            bank_interfered = phase2d_multiply(projected[bank_name], weight)
            interfered.append(bank_interfered)
        
        # 3. Sum interference patterns (constructive + destructive)
        combined = torch.stack(interfered, dim=0).sum(dim=0)  # [batch, seq, dim, 2]
        
        # 4. Cross-bank interaction
        # Concatenate all banks and project
        if len(available_banks) == self.num_banks:
            all_banks = torch.cat([projected[name] for name in self.bank_names], dim=-2)
            cross = self.cross_proj(all_banks)
            combined = combined + cross * 0.1
        
        # 5. Output fusion
        output = self.norm(combined)
        output = self.fusion(output)
        output = self.dropout(output[..., 0]).unsqueeze(-1).expand_as(output) * output + output
        
        return output
    
    def compute_coupling_loss(
        self,
        bank_outputs: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Compute cross-bank coherence loss.
        
        Encourages phase alignment between banks (when appropriate).
        """
        available_banks = [name for name in self.bank_names if name in bank_outputs]
        if len(available_banks) < 2:
            return None
        
        # IMPORTANT (memory): computing coherence over full [B, S, D, 2] for many bank-pairs
        # can OOM at medium/large sizes. We keep this loss cheap by reducing sequence length
        # before calling phase2d_coherence.
        #
        # Strategy: subsample up to K positions, then mean-pool over sequence -> [B, D, 2].
        # This makes phase2d_coherence operate on [B, D, 2] instead of [B, S, D, 2].
        max_tokens_for_loss = 64
        
        # Compute pairwise coherence between banks
        coherence_loss = 0.0
        num_pairs = 0
        
        for i, name_i in enumerate(available_banks):
            for name_j in available_banks[i+1:]:
                bank_i = bank_outputs[name_i]
                bank_j = bank_outputs[name_j]
                
                # Reduce sequence dimension for stability + memory
                # bank_*: [B, S, D, 2] -> [B, D, 2]
                if bank_i.dim() == 4:
                    seq_len = bank_i.shape[1]
                    if seq_len > max_tokens_for_loss:
                        idx = torch.linspace(
                            0,
                            seq_len - 1,
                            steps=max_tokens_for_loss,
                            device=bank_i.device,
                        ).long()
                        bank_i = bank_i.index_select(dim=1, index=idx)
                        bank_j = bank_j.index_select(dim=1, index=idx)
                    bank_i = bank_i.mean(dim=1)
                    bank_j = bank_j.mean(dim=1)
                
                # Coherence: normalized dot product (real part)
                # We want banks to have SOME coherence (not random)
                coherence = phase2d_coherence(bank_i, bank_j)
                
                # Loss: encourage positive coherence (0 to 1)
                # We use -coherence so minimizing loss increases coherence
                coherence_loss = coherence_loss - coherence.mean()
                num_pairs += 1
        
        if num_pairs > 0:
            coherence_loss = coherence_loss / num_pairs
        
        return coherence_loss
