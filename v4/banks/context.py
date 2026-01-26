#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Phase Bank: Local composition and syntax-like modulation

Captures how tokens interact and modify each other based on context.
Uses local phase interference patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from ..core.interfaces import PhaseBank
from ..core.registry import register_bank
from ..core.phase2d import (
    Phase2DLinear, Phase2DLayerNorm, 
    phase2d_multiply, phase2d_normalize, phase2d_coherence
)


@register_bank('context', description='Context/syntax modulation via local phase interference')
class ContextPhaseBank(nn.Module):
    """
    Context Phase Bank: captures local compositional structure.
    
    Key features:
    - Local phase interference (neighbor interactions)
    - Learned phase modulation based on context
    - No global attention (linear complexity)
    """
    
    def __init__(
        self,
        dim: int = 256,
        window_size: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._dim = dim
        self.window_size = window_size
        self.num_layers = num_layers
        
        # Local context extraction (causal convolution-like)
        self.context_proj = Phase2DLinear(dim, dim)
        
        # Phase modulation based on context
        self.modulation_proj = Phase2DLinear(dim * 2, dim)
        
        # Processing layers
        self.layers = nn.ModuleList([
            Phase2DLinear(dim, dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            Phase2DLayerNorm(dim) for _ in range(num_layers)
        ])
        
        # Learned window weights (how much each neighbor contributes)
        self.window_weights = nn.Parameter(torch.ones(window_size) / window_size)
        
        # Output
        self.output_proj = Phase2DLinear(dim, dim)
        self.output_norm = Phase2DLayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def name(self) -> str:
        return "context"
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def _compute_local_context(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute local context via causal windowed aggregation.
        
        Args:
            x: [batch, seq, dim, 2] Phase2D input
        
        Returns:
            [batch, seq, dim, 2] local context
        """
        batch_size, seq_len, dim, _ = x.shape
        
        # Pad for causal context
        # We only look at past tokens (causal)
        padded = F.pad(x, (0, 0, 0, 0, self.window_size - 1, 0), mode='constant', value=0)
        
        # Compute weighted sum over window (unfold-based for efficiency)
        # [batch, seq, dim, 2] -> [batch, seq, window_size, dim, 2]
        windows = padded.unfold(1, self.window_size, 1)  # [batch, seq, dim, 2, window]
        windows = windows.permute(0, 1, 4, 2, 3)  # [batch, seq, window, dim, 2]
        
        # Apply window weights
        weights = F.softmax(self.window_weights, dim=0)  # [window]
        weights = weights.view(1, 1, self.window_size, 1, 1)
        
        # Weighted sum (phase interference)
        context = (windows * weights).sum(dim=2)  # [batch, seq, dim, 2]
        
        return context
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, dim, 2]
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Process input through context phase bank.
        
        Args:
            x: Phase2D input embeddings [batch, seq, dim, 2]
            context: Optional context (not used in this bank)
        
        Returns:
            Phase2D context representation [batch, seq, dim, 2]
        """
        # 1. Compute local context
        local_ctx = self._compute_local_context(x)
        local_ctx = self.context_proj(local_ctx)
        
        # 2. Phase modulation: modulate current token by context
        # Concatenate current and context, then project
        combined = torch.cat([x, local_ctx], dim=-2)  # [batch, seq, dim*2, 2]
        modulation = self.modulation_proj(combined)  # [batch, seq, dim, 2]
        
        # Apply modulation via complex multiplication
        h = phase2d_multiply(x, phase2d_normalize(modulation))
        
        # 3. Process through layers
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = norm(h)
            h = layer(h)
            h = F.gelu(h[..., 0]).unsqueeze(-1) * h
            h = residual + h * 0.1
        
        # 4. Output
        out = self.output_norm(h)
        out = self.output_proj(out)
        
        return out
