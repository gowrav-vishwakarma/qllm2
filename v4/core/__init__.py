#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Core - Quantum Phase-Field LLM
Injectable plugin-style architecture with Phase2D representation
"""

from .phase2d import (
    Phase2D, 
    phase2d_from_real, phase2d_to_real, phase2d_multiply, phase2d_rotate,
    phase2d_magnitude, phase2d_normalize, phase2d_conjugate, phase2d_coherence,
    phase2d_apply_iota,
    IotaBlock, Phase2DEmbed, Phase2DLinear, Phase2DLayerNorm,
)
from .interfaces import PhaseBank, Coupler, Backbone, Memory, Objective, Sampler, BackboneState
from .registry import Registry, get_registry
from .config import V4Config, load_config, get_default_config
from .morphology_embed import (
    MorphologyAwareEmbed,
    AffixRotationBank,
    DualEmbedding,
)

__all__ = [
    # Phase2D math
    'Phase2D', 'phase2d_from_real', 'phase2d_to_real', 'phase2d_multiply', 'phase2d_rotate',
    'phase2d_magnitude', 'phase2d_normalize', 'phase2d_conjugate', 'phase2d_coherence',
    'phase2d_apply_iota',
    'IotaBlock', 'Phase2DEmbed', 'Phase2DLinear', 'Phase2DLayerNorm',
    # Morphology embedding
    'MorphologyAwareEmbed', 'AffixRotationBank', 'DualEmbedding',
    # Interfaces
    'PhaseBank', 'Coupler', 'Backbone', 'Memory', 'Objective', 'Sampler', 'BackboneState',
    # Registry
    'Registry', 'get_registry',
    # Config
    'V4Config', 'load_config', 'get_default_config',
]
