#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MorphologyPhaseBank: Grammatical transformation layer

Focuses on morphological structure - how word forms encode grammatical
information (tense, aspect, case, number, etc.) in the phase space.

When used with morphological tokenizer, this bank can leverage the
explicit root/affix decomposition. When used with BPE, it learns to
infer morphological patterns from subword structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from ..core.interfaces import PhaseBank
from ..core.registry import register_bank
from ..core.phase2d import (
    Phase2DLinear, Phase2DLayerNorm, IotaBlock,
    phase2d_multiply, phase2d_normalize, phase2d_magnitude,
)


@register_bank('morphology', description='Morphological structure layer for grammatical transformations')
class MorphologyPhaseBank(nn.Module):
    """
    Morphology Phase Bank: represents grammatical transformations in phase space.
    
    Key features:
    - Learns morphological patterns from token sequences
    - Models grammatical transformations as phase rotations
    - When affix information is available, uses it to guide learning
    
    The bank processes sequences to extract morphological features and
    applies learned transformations that encode grammatical relationships.
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_patterns: int = 128,  # Number of learned morphological patterns
        num_layers: int = 2,
        use_affix_info: bool = True,  # Whether to use prefix/suffix IDs if available
        dropout: float = 0.1,
    ):
        super().__init__()
        self._dim = dim
        self.num_patterns = num_patterns
        self.num_layers = num_layers
        self.use_affix_info = use_affix_info
        
        # Morphological pattern embeddings
        # These represent abstract grammatical patterns (tense, case, etc.)
        self.pattern_bank = nn.Parameter(torch.randn(num_patterns, dim, 2) * 0.02)
        
        # Pattern detection: identify which morphological pattern is active
        self.pattern_detector = Phase2DLinear(dim, num_patterns)
        
        # Transformation layers for applying morphological patterns
        self.transform_layers = nn.ModuleList([
            nn.ModuleDict({
                'rotation': IotaBlock(dim),
                'linear': Phase2DLinear(dim, dim),
                'norm': Phase2DLayerNorm(dim),
            })
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = Phase2DLinear(dim, dim)
        self.output_norm = Phase2DLayerNorm(dim)
        
        # If using affix info, project from affix space
        if use_affix_info:
            # These will be set based on actual affix vocab sizes
            self.prefix_proj = nn.Linear(1, dim)  # Placeholder
            self.suffix_proj = nn.Linear(1, dim)  # Placeholder
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def name(self) -> str:
        return "morphology"
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, dim, 2]
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Process input through morphology phase bank.
        
        Args:
            x: Phase2D input embeddings [batch, seq, dim, 2]
            context: Optional context with 'prefix_ids', 'suffix_ids' for morphological mode
        
        Returns:
            Phase2D morphological representation [batch, seq, dim, 2]
        """
        batch_size, seq_len, dim, _ = x.shape
        context = context or {}
        
        # 1. Detect morphological patterns in input
        pattern_query = self.pattern_detector(x)  # [batch, seq, num_patterns, 2]
        
        # Compute pattern attention using magnitude
        pattern_mag = phase2d_magnitude(pattern_query)  # [batch, seq, num_patterns]
        pattern_attn = F.softmax(pattern_mag, dim=-1)  # [batch, seq, num_patterns]
        
        # Retrieve pattern embeddings
        # [batch, seq, num_patterns] @ [num_patterns, dim, 2] -> [batch, seq, dim, 2]
        pattern_embed = torch.einsum('bsn,ndp->bsdp', pattern_attn, self.pattern_bank)
        
        # 2. Inject morphological patterns into representation
        h = x + pattern_embed * 0.1
        
        # 3. Apply transformation layers
        for layer_dict in self.transform_layers:
            residual = h
            
            # Normalize
            h = layer_dict['norm'](h)
            
            # Apply learned rotation (morphological transformation)
            h = layer_dict['rotation'](h)
            
            # Linear transformation
            h = layer_dict['linear'](h)
            
            # Activation (GELU on real part, preserving phase structure)
            h_real = F.gelu(h[..., 0])
            h_imag = h[..., 1] * torch.sigmoid(h[..., 0])  # Phase-aware activation
            h = torch.stack([h_real, h_imag], dim=-1)
            
            # Dropout and residual
            h = self.dropout(h[..., 0]).unsqueeze(-1).expand_as(h) * h + h
            h = residual + h * 0.1
        
        # 4. Output projection
        out = self.output_norm(h)
        out = self.output_proj(out)
        
        return out
    
    def get_pattern_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the morphological pattern distribution for each token.
        
        Useful for analysis and interpretability.
        
        Args:
            x: Phase2D input [batch, seq, dim, 2]
        
        Returns:
            Pattern probabilities [batch, seq, num_patterns]
        """
        pattern_query = self.pattern_detector(x)
        pattern_mag = phase2d_magnitude(pattern_query)
        return F.softmax(pattern_mag, dim=-1)
