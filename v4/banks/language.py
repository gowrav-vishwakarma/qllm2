#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language Phase Bank: Cross-lingual and morphological representation

Captures language-specific patterns and supports multilingual processing.
Can be gated based on language ID.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from ..core.interfaces import PhaseBank
from ..core.registry import register_bank
from ..core.phase2d import Phase2DLinear, Phase2DLayerNorm, phase2d_multiply


@register_bank('language', description='Language-specific phase modulation for multilingual support')
class LanguagePhaseBank(nn.Module):
    """
    Language Phase Bank: language-specific representations.
    
    Key features:
    - Language-specific phase gates
    - Morphological pattern encoding
    - Cross-lingual alignment support
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_languages: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._dim = dim
        self.num_languages = num_languages
        self.num_layers = num_layers
        
        # Language-specific phase gates (one per language)
        # Each gate is a complex scaling factor
        self.language_gates = nn.Parameter(torch.randn(num_languages, dim, 2) * 0.02)
        # Initialize close to identity (real=1, imag=0)
        self.language_gates.data[..., 0] = 1.0
        self.language_gates.data[..., 1] = 0.0
        
        # Universal language-agnostic processing
        self.universal_proj = Phase2DLinear(dim, dim)
        
        # Language-specific projections
        self.language_projs = nn.ModuleList([
            Phase2DLinear(dim, dim) for _ in range(num_languages)
        ])
        
        # Processing layers
        self.layers = nn.ModuleList([
            Phase2DLinear(dim, dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            Phase2DLayerNorm(dim) for _ in range(num_layers)
        ])
        
        # Output
        self.output_proj = Phase2DLinear(dim, dim)
        self.output_norm = Phase2DLayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def name(self) -> str:
        return "language"
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, dim, 2]
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Process input through language phase bank.
        
        Args:
            x: Phase2D input embeddings [batch, seq, dim, 2]
            context: Optional context with 'language_id' (int or tensor)
        
        Returns:
            Phase2D language representation [batch, seq, dim, 2]
        """
        batch_size, seq_len, dim, _ = x.shape
        
        # Get language ID from context (default to 0 = English/universal)
        language_id = 0
        if context is not None and 'language_id' in context:
            language_id = context['language_id']
            if isinstance(language_id, torch.Tensor):
                language_id = language_id.item()
        
        language_id = min(language_id, self.num_languages - 1)
        
        # 1. Apply language-specific gate (phase modulation)
        gate = self.language_gates[language_id]  # [dim, 2]
        gate = gate.unsqueeze(0).unsqueeze(0)  # [1, 1, dim, 2]
        
        h = phase2d_multiply(x, gate)
        
        # 2. Universal processing
        h_universal = self.universal_proj(h)
        
        # 3. Language-specific processing
        h_specific = self.language_projs[language_id](h)
        
        # 4. Combine universal and specific
        h = h_universal + h_specific * 0.5
        
        # 5. Process through layers
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = norm(h)
            h = layer(h)
            h = F.gelu(h[..., 0]).unsqueeze(-1) * h
            h = residual + h * 0.1
        
        # 6. Output
        out = self.output_norm(h)
        out = self.output_proj(out)
        
        return out


@register_bank('emotion', description='Emotional tone phase bank (placeholder)')
class EmotionPhaseBank(nn.Module):
    """
    Emotion Phase Bank: emotional tone representation.
    
    Placeholder implementation - can be enhanced with:
    - Sentiment classifiers
    - Valence-arousal phase encoding
    - Emotion-specific interference patterns
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_emotions: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._dim = dim
        self.num_emotions = num_emotions
        
        # Emotion embeddings (phase-coded)
        self.emotion_embeds = nn.Parameter(torch.randn(num_emotions, dim, 2) * 0.02)
        
        # Emotion detection (from phase representation)
        self.emotion_detector = Phase2DLinear(dim, num_emotions)
        
        # Processing
        self.proc = Phase2DLinear(dim, dim)
        self.norm = Phase2DLayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def name(self) -> str:
        return "emotion"
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Minimal emotion processing (placeholder)"""
        batch_size, seq_len, dim, _ = x.shape
        
        # Detect emotion (soft attention over emotion embeddings)
        emotion_logits = self.emotion_detector(x)[..., 0]  # [batch, seq, num_emotions]
        emotion_weights = F.softmax(emotion_logits, dim=-1)
        
        # Retrieve emotion representation
        emotion_repr = torch.einsum('bsn,ndp->bsdp', emotion_weights, self.emotion_embeds)
        
        # Modulate input with emotion
        h = x + emotion_repr * 0.1
        h = self.norm(h)
        h = self.proc(h)
        
        return h
