#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrthographyPhaseBank: Script/shape pattern layer

Learns orthographic patterns from subword form:
- Word boundaries (start/end of word markers)
- Case patterns (capitalization)
- Script features (digit, punctuation, letter classes)
- Character shape patterns

This helps with:
- Multilingual generalization (scripts share shape patterns)
- Stabilizing morphology learning (provides surface form cues)
- Handling out-of-vocabulary tokens gracefully
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from ..core.interfaces import PhaseBank
from ..core.registry import register_bank
from ..core.phase2d import (
    Phase2DLinear, Phase2DLayerNorm, IotaBlock,
    phase2d_magnitude, phase2d_normalize,
)


# Orthographic feature categories
ORTHO_FEATURES = {
    'is_word_start': 0,
    'is_word_end': 1,
    'is_capitalized': 2,
    'is_all_caps': 3,
    'is_digit': 4,
    'is_punctuation': 5,
    'is_mixed_case': 6,
    'has_hyphen': 7,
}
NUM_ORTHO_FEATURES = len(ORTHO_FEATURES)


def extract_orthographic_features(token_ids: torch.Tensor, tokenizer=None) -> torch.Tensor:
    """
    Extract orthographic features from token IDs.
    
    If tokenizer is not provided, returns zeros (features will be learned implicitly).
    
    Args:
        token_ids: [batch, seq] token indices
        tokenizer: Optional tokenizer to decode tokens
    
    Returns:
        [batch, seq, NUM_ORTHO_FEATURES] feature tensor
    """
    batch_size, seq_len = token_ids.shape
    device = token_ids.device
    
    # Default: return zeros (implicit learning)
    # In production, you would decode tokens and extract actual features
    features = torch.zeros(batch_size, seq_len, NUM_ORTHO_FEATURES, device=device)
    
    # Heuristic: assume word boundaries at special positions
    # (This is a simplified version - real implementation would use tokenizer)
    
    return features


@register_bank('orthography', description='Orthographic pattern layer for script/shape features')
class OrthographyPhaseBank(nn.Module):
    """
    Orthography Phase Bank: represents script and shape patterns in phase space.
    
    Key features:
    - Learns character/subword shape patterns
    - Encodes word boundary information
    - Captures case and script features
    - Helps multilingual generalization
    
    The bank provides a stable representation of surface form that
    complements semantic and morphological information.
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_shape_patterns: int = 64,  # Learned shape patterns
        num_script_classes: int = 16,   # Script/writing system categories
        num_layers: int = 2,
        use_explicit_features: bool = False,  # Whether to use extracted features
        dropout: float = 0.1,
    ):
        super().__init__()
        self._dim = dim
        self.num_shape_patterns = num_shape_patterns
        self.num_script_classes = num_script_classes
        self.num_layers = num_layers
        self.use_explicit_features = use_explicit_features
        
        # Shape pattern embeddings (learned visual/orthographic patterns)
        self.shape_patterns = nn.Parameter(torch.randn(num_shape_patterns, dim, 2) * 0.02)
        
        # Script class embeddings (writing system categories)
        self.script_embeddings = nn.Parameter(torch.randn(num_script_classes, dim, 2) * 0.02)
        
        # Pattern detection: identify shape patterns from Phase2D input
        self.shape_detector = Phase2DLinear(dim, num_shape_patterns)
        
        # Script classifier: soft assignment to script classes
        self.script_classifier = Phase2DLinear(dim, num_script_classes)
        
        # Explicit feature projection (if enabled)
        if use_explicit_features:
            self.feature_proj = nn.Linear(NUM_ORTHO_FEATURES, dim)
        
        # Processing layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'linear': Phase2DLinear(dim, dim),
                'norm': Phase2DLayerNorm(dim),
                'gate': nn.Linear(dim * 2, dim),  # Gating for orthographic modulation
            })
            for _ in range(num_layers)
        ])
        
        # Positional encoding for word-internal position
        # (helps distinguish word-start, word-middle, word-end)
        self.position_encoding = nn.Parameter(torch.randn(1, 1, dim, 2) * 0.02)
        
        # Output projection
        self.output_proj = Phase2DLinear(dim, dim)
        self.output_norm = Phase2DLayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def name(self) -> str:
        return "orthography"
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, dim, 2]
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Process input through orthography phase bank.
        
        Args:
            x: Phase2D input embeddings [batch, seq, dim, 2]
            context: Optional context (may contain tokenizer for feature extraction)
        
        Returns:
            Phase2D orthographic representation [batch, seq, dim, 2]
        """
        batch_size, seq_len, dim, _ = x.shape
        context = context or {}
        
        # 1. Detect shape patterns
        shape_query = self.shape_detector(x)  # [batch, seq, num_shapes, 2]
        shape_mag = phase2d_magnitude(shape_query)
        shape_attn = F.softmax(shape_mag, dim=-1)  # [batch, seq, num_shapes]
        
        # Retrieve shape embeddings
        shape_embed = torch.einsum('bsn,ndp->bsdp', shape_attn, self.shape_patterns)
        
        # 2. Classify script/writing system
        script_query = self.script_classifier(x)  # [batch, seq, num_scripts, 2]
        script_mag = phase2d_magnitude(script_query)
        script_attn = F.softmax(script_mag, dim=-1)  # [batch, seq, num_scripts]
        
        # Retrieve script embeddings
        script_embed = torch.einsum('bsn,ndp->bsdp', script_attn, self.script_embeddings)
        
        # 3. Combine orthographic information
        h = x + shape_embed * 0.1 + script_embed * 0.05
        
        # 4. Add explicit features if enabled
        if self.use_explicit_features:
            token_ids = context.get('token_ids')
            if token_ids is not None:
                tokenizer = context.get('tokenizer')
                features = extract_orthographic_features(token_ids, tokenizer)
                feature_embed = self.feature_proj(features)  # [batch, seq, dim]
                # Add to real part of Phase2D
                h[..., 0] = h[..., 0] + feature_embed * 0.1
        
        # 5. Apply processing layers
        for layer_dict in self.layers:
            residual = h
            
            # Normalize
            h = layer_dict['norm'](h)
            
            # Linear transformation
            h = layer_dict['linear'](h)
            
            # Compute gate from real representation
            h_flat = torch.cat([h[..., 0], h[..., 1]], dim=-1)  # [batch, seq, dim*2]
            gate = torch.sigmoid(layer_dict['gate'](h_flat))  # [batch, seq, dim]
            
            # Apply gate
            h = h * gate.unsqueeze(-1)
            
            # Dropout and residual
            h = self.dropout(h[..., 0]).unsqueeze(-1).expand_as(h) * h + h
            h = residual + h * 0.1
        
        # 6. Output projection
        out = self.output_norm(h)
        out = self.output_proj(out)
        
        return out
    
    def get_shape_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the shape pattern distribution for each token.
        
        Args:
            x: Phase2D input [batch, seq, dim, 2]
        
        Returns:
            Shape pattern probabilities [batch, seq, num_shape_patterns]
        """
        shape_query = self.shape_detector(x)
        shape_mag = phase2d_magnitude(shape_query)
        return F.softmax(shape_mag, dim=-1)
    
    def get_script_distribution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the script class distribution for each token.
        
        Args:
            x: Phase2D input [batch, seq, dim, 2]
        
        Returns:
            Script class probabilities [batch, seq, num_script_classes]
        """
        script_query = self.script_classifier(x)
        script_mag = phase2d_magnitude(script_query)
        return F.softmax(script_mag, dim=-1)
