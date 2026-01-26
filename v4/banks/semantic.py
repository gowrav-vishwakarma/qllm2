#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Phase Bank: Core semantic/meaning layer

Captures the inherent meaning of tokens in phase space.
Provides the foundation for concept-level understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from ..core.interfaces import PhaseBank
from ..core.registry import register_bank
from ..core.phase2d import Phase2DLinear, Phase2DLayerNorm, phase2d_normalize


@register_bank('semantic', description='Semantic meaning layer with concept supervision')
class SemanticPhaseBank(nn.Module):
    """
    Semantic Phase Bank: represents token semantics in phase space.
    
    Key features:
    - Learnable concept embeddings
    - Phase-based semantic similarity
    - Concept supervision anchor for training
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_concepts: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._dim = dim
        self.num_concepts = num_concepts
        self.num_layers = num_layers
        
        # Concept memory: learnable phase-coded concepts
        # Each concept is a Phase2D vector
        self.concept_memory = nn.Parameter(torch.randn(num_concepts, dim, 2) * 0.02)
        
        # Project input to concept space
        self.concept_proj = Phase2DLinear(dim, num_concepts)
        
        # Semantic processing layers
        self.layers = nn.ModuleList([
            Phase2DLinear(dim, dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            Phase2DLayerNorm(dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = Phase2DLinear(dim, dim)
        self.output_norm = Phase2DLayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def name(self) -> str:
        return "semantic"
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, dim, 2]
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Process input through semantic phase bank.
        
        Args:
            x: Phase2D input embeddings [batch, seq, dim, 2]
            context: Optional context with 'token_ids' for concept supervision
        
        Returns:
            Phase2D semantic representation [batch, seq, dim, 2]
        """
        batch_size, seq_len, dim, _ = x.shape
        
        # 1. Compute concept attention via phase coherence
        # Project to concept space
        concept_query = self.concept_proj(x)  # [batch, seq, num_concepts, 2]
        
        # Compute coherence with concept memory (no trig!)
        # coherence = Re(query * conj(memory)) / (|query| * |memory|)
        concept_memory = self.concept_memory.unsqueeze(0).unsqueeze(0)  # [1, 1, num_concepts, dim, 2]
        
        # Simple attention: use real part of dot product
        query_real = concept_query[..., 0]  # [batch, seq, num_concepts]
        memory_real = self.concept_memory[..., 0].mean(dim=-1)  # [num_concepts]
        
        concept_attn = F.softmax(query_real + memory_real.unsqueeze(0).unsqueeze(0), dim=-1)
        
        # Retrieve from concept memory
        # [batch, seq, num_concepts] @ [num_concepts, dim, 2] -> [batch, seq, dim, 2]
        concept_real = torch.einsum('bsn,ndp->bsdp', concept_attn, self.concept_memory)
        
        # 2. Process through semantic layers
        h = x + concept_real * 0.1  # Soft concept injection
        
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = norm(h)
            h = layer(h)
            h = F.gelu(h[..., 0]).unsqueeze(-1) * h  # GELU on magnitude
            h = self.dropout(h[..., 0]).unsqueeze(-1).expand_as(h) * h + h  # Dropout
            h = residual + h * 0.1
        
        # 3. Output projection
        out = self.output_norm(h)
        out = self.output_proj(out)
        
        return out
    
    def get_concept_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get concept logits for supervision (used by concept loss).
        
        Args:
            x: Phase2D semantic output [batch, seq, dim, 2]
        
        Returns:
            Concept logits [batch, seq, num_concepts]
        """
        # Project to concept space and return real part as logits
        concept_proj = self.concept_proj(x)
        return concept_proj[..., 0]  # Real part as logits
