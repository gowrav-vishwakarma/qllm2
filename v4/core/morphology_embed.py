#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MorphologyAwareEmbed: Phase2D embedding with root + affix rotation

Implements the user's idea of morphological embedding:
- Root sets the base Phase2D vector
- Prefix selects a learned rotation operator (applied before context)
- Suffix selects a learned rotation operator (applied after context)

Math:
    z = EmbedRoot(root_id)
    z = RotatePrefix(prefix_id) ⊙ z  (complex multiplication = rotation)
    z = RotateSuffix(suffix_id) ⊙ z

This allows prefixes and suffixes to apply semantic/grammatical transformations
as phase rotations, preserving the magnitude (core meaning) while shifting
the phase (relational/contextual meaning).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from .phase2d import (
    Phase2DEmbed, 
    IotaBlock, 
    phase2d_multiply,
    phase2d_normalize,
)


class AffixRotationBank(nn.Module):
    """
    Bank of learnable rotation operators for affixes (prefixes or suffixes).
    
    Each affix maps to a rotation in Phase2D space via IotaBlock (Cayley transform).
    Applying an affix = multiplying by its rotation (complex multiplication).
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        null_id: int = 4,  # ID for null affix (no rotation)
    ):
        """
        Args:
            vocab_size: Size of affix vocabulary
            dim: Phase dimension
            null_id: ID of the null/identity affix
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.null_id = null_id
        
        # Learnable rotation parameters per affix
        # Each affix has a rotation angle parameterized via Cayley transform
        self.rotation_params = nn.Parameter(torch.randn(vocab_size, dim) * 0.01)
        
        # Initialize null affix to identity (zero rotation)
        if null_id < vocab_size:
            with torch.no_grad():
                self.rotation_params[null_id].zero_()
    
    def get_rotation(self, affix_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotation components (cos-like, sin-like) for affix IDs.
        
        Uses Cayley transform: cos = (1 - a²)/(1 + a²), sin = 2a/(1 + a²)
        
        Args:
            affix_ids: [batch, seq] affix token IDs
        
        Returns:
            (cos_like, sin_like): [batch, seq, dim] rotation components
        """
        # Get rotation parameters for each affix
        a = self.rotation_params[affix_ids]  # [batch, seq, dim]
        
        # Cayley transform for stable rotation without trig
        a_sq = a * a
        denom = 1.0 + a_sq
        
        cos_like = (1.0 - a_sq) / denom  # [batch, seq, dim]
        sin_like = (2.0 * a) / denom      # [batch, seq, dim]
        
        return cos_like, sin_like
    
    def forward(
        self, 
        x: torch.Tensor,  # [batch, seq, dim, 2]
        affix_ids: torch.Tensor,  # [batch, seq]
    ) -> torch.Tensor:
        """
        Apply affix rotations to Phase2D vectors.
        
        Args:
            x: Phase2D embeddings [batch, seq, dim, 2]
            affix_ids: Affix token IDs [batch, seq]
        
        Returns:
            Rotated Phase2D embeddings [batch, seq, dim, 2]
        """
        cos_like, sin_like = self.get_rotation(affix_ids)
        
        # Apply rotation: complex multiplication by (cos + i*sin)
        x_real = x[..., 0]  # [batch, seq, dim]
        x_imag = x[..., 1]  # [batch, seq, dim]
        
        out_real = x_real * cos_like - x_imag * sin_like
        out_imag = x_real * sin_like + x_imag * cos_like
        
        return torch.stack([out_real, out_imag], dim=-1)


class MorphologyAwareEmbed(nn.Module):
    """
    Morphology-aware Phase2D embedding.
    
    Takes (root_ids, prefix_ids, suffix_ids) and produces Phase2D embeddings
    where:
    1. Root provides the base semantic vector
    2. Prefix applies a pre-rotation (modifies meaning category)
    3. Suffix applies a post-rotation (modifies grammatical role)
    
    The rotations are learned end-to-end through the Cayley parameterization.
    """
    
    def __init__(
        self,
        root_vocab_size: int,
        prefix_vocab_size: int,
        suffix_vocab_size: int,
        dim: int,
        null_affix_id: int = 4,  # Default null affix ID
        padding_idx: int = 0,
    ):
        """
        Args:
            root_vocab_size: Size of root vocabulary
            prefix_vocab_size: Size of prefix vocabulary
            suffix_vocab_size: Size of suffix vocabulary
            dim: Phase dimension
            null_affix_id: ID for null/identity affix
            padding_idx: Padding token ID
        """
        super().__init__()
        self.dim = dim
        self.root_vocab_size = root_vocab_size
        self.prefix_vocab_size = prefix_vocab_size
        self.suffix_vocab_size = suffix_vocab_size
        self.null_affix_id = null_affix_id
        
        # Root embedding (Phase2D)
        self.root_embed = Phase2DEmbed(
            vocab_size=root_vocab_size,
            dim=dim,
            padding_idx=padding_idx,
        )
        
        # Prefix rotation bank
        self.prefix_rotations = AffixRotationBank(
            vocab_size=prefix_vocab_size,
            dim=dim,
            null_id=null_affix_id,
        )
        
        # Suffix rotation bank
        self.suffix_rotations = AffixRotationBank(
            vocab_size=suffix_vocab_size,
            dim=dim,
            null_id=null_affix_id,
        )
        
        # Optional: learned magnitude modulation for affixes
        # (affixes can also scale magnitude, not just rotate)
        self.prefix_magnitude = nn.Embedding(prefix_vocab_size, dim)
        self.suffix_magnitude = nn.Embedding(suffix_vocab_size, dim)
        
        # Initialize magnitude modulation near 1.0 (no change)
        nn.init.constant_(self.prefix_magnitude.weight, 1.0)
        nn.init.constant_(self.suffix_magnitude.weight, 1.0)
        
        # Set null affix magnitudes to exactly 1.0
        with torch.no_grad():
            if null_affix_id < prefix_vocab_size:
                self.prefix_magnitude.weight[null_affix_id].fill_(1.0)
            if null_affix_id < suffix_vocab_size:
                self.suffix_magnitude.weight[null_affix_id].fill_(1.0)
    
    def forward(
        self,
        root_ids: torch.Tensor,    # [batch, seq]
        prefix_ids: torch.Tensor,  # [batch, seq]
        suffix_ids: torch.Tensor,  # [batch, seq]
        apply_magnitude: bool = True,
    ) -> torch.Tensor:
        """
        Compute morphology-aware Phase2D embeddings.
        
        Args:
            root_ids: Root token IDs [batch, seq]
            prefix_ids: Prefix token IDs [batch, seq]
            suffix_ids: Suffix token IDs [batch, seq]
            apply_magnitude: Whether to apply magnitude modulation
        
        Returns:
            Phase2D embeddings [batch, seq, dim, 2]
        """
        # 1. Get base root embedding
        z = self.root_embed(root_ids)  # [batch, seq, dim, 2]
        
        # 2. Apply prefix rotation
        z = self.prefix_rotations(z, prefix_ids)
        
        # 3. Apply prefix magnitude modulation (optional)
        if apply_magnitude:
            prefix_mag = self.prefix_magnitude(prefix_ids)  # [batch, seq, dim]
            # Soft modulation: 0.5 + 0.5 * sigmoid for stability
            prefix_mag = 0.5 + 0.5 * torch.sigmoid(prefix_mag - 1.0)
            z = z * prefix_mag.unsqueeze(-1)
        
        # 4. Apply suffix rotation
        z = self.suffix_rotations(z, suffix_ids)
        
        # 5. Apply suffix magnitude modulation (optional)
        if apply_magnitude:
            suffix_mag = self.suffix_magnitude(suffix_ids)  # [batch, seq, dim]
            suffix_mag = 0.5 + 0.5 * torch.sigmoid(suffix_mag - 1.0)
            z = z * suffix_mag.unsqueeze(-1)
        
        return z
    
    def get_affix_coherence(
        self,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute coherence between prefix and suffix rotations.
        
        High coherence = similar transformations (may indicate redundancy)
        Low coherence = complementary transformations (desirable)
        
        Args:
            prefix_ids: [batch, seq] prefix IDs
            suffix_ids: [batch, seq] suffix IDs
        
        Returns:
            Coherence score [batch, seq]
        """
        prefix_cos, prefix_sin = self.prefix_rotations.get_rotation(prefix_ids)
        suffix_cos, suffix_sin = self.suffix_rotations.get_rotation(suffix_ids)
        
        # Dot product of rotation vectors
        dot = (prefix_cos * suffix_cos + prefix_sin * suffix_sin).mean(dim=-1)
        
        return dot


class DualEmbedding(nn.Module):
    """
    Dual embedding module that supports both BPE and Morphological modes.
    
    Allows switching between:
    - Standard Phase2DEmbed (for BPE tokenizer)
    - MorphologyAwareEmbed (for Morphological tokenizer)
    
    This enables A/B testing between tokenization strategies.
    """
    
    def __init__(
        self,
        bpe_vocab_size: int = 50257,
        root_vocab_size: int = 16000,
        prefix_vocab_size: int = 2000,
        suffix_vocab_size: int = 2000,
        dim: int = 256,
        mode: str = 'bpe',  # 'bpe' or 'morphological'
    ):
        """
        Args:
            bpe_vocab_size: Vocabulary size for BPE mode
            root_vocab_size: Root vocab size for morphological mode
            prefix_vocab_size: Prefix vocab size for morphological mode
            suffix_vocab_size: Suffix vocab size for morphological mode
            dim: Phase dimension
            mode: Initial mode ('bpe' or 'morphological')
        """
        super().__init__()
        self.dim = dim
        self._mode = mode
        
        # BPE embedding
        self.bpe_embed = Phase2DEmbed(
            vocab_size=bpe_vocab_size,
            dim=dim,
            padding_idx=0,
        )
        
        # Morphological embedding
        self.morph_embed = MorphologyAwareEmbed(
            root_vocab_size=root_vocab_size,
            prefix_vocab_size=prefix_vocab_size,
            suffix_vocab_size=suffix_vocab_size,
            dim=dim,
        )
    
    @property
    def mode(self) -> str:
        return self._mode
    
    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ['bpe', 'morphological']:
            raise ValueError(f"Mode must be 'bpe' or 'morphological', got {value}")
        self._mode = value
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        root_ids: Optional[torch.Tensor] = None,
        prefix_ids: Optional[torch.Tensor] = None,
        suffix_ids: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute embeddings based on mode.
        
        Args:
            input_ids: BPE token IDs [batch, seq] (for BPE mode)
            root_ids: Root IDs [batch, seq] (for morphological mode)
            prefix_ids: Prefix IDs [batch, seq] (for morphological mode)
            suffix_ids: Suffix IDs [batch, seq] (for morphological mode)
            mode: Override default mode for this call
        
        Returns:
            Phase2D embeddings [batch, seq, dim, 2]
        """
        use_mode = mode or self._mode
        
        if use_mode == 'bpe':
            if input_ids is None:
                raise ValueError("input_ids required for BPE mode")
            return self.bpe_embed(input_ids)
        
        elif use_mode == 'morphological':
            if root_ids is None or prefix_ids is None or suffix_ids is None:
                raise ValueError("root_ids, prefix_ids, suffix_ids required for morphological mode")
            return self.morph_embed(root_ids, prefix_ids, suffix_ids)
        
        else:
            raise ValueError(f"Unknown mode: {use_mode}")
