#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Quantum Phase-Field LLM

A novel architecture combining:
- Phase2D representation (complex/phase without trig)
- Multi-layer phase banks (semantic/context/language/emotion)
- Oscillatory SSM backbone (linear-time)
- Interference coupling
- Phase-coded associative memory
- Incremental learning support

All operations are GPU-friendly (GEMM-based, Tensor Core optimized).
"""

from .core import (
    Phase2D, Phase2DEmbed, Phase2DLinear, Phase2DLayerNorm, IotaBlock,
    phase2d_multiply, phase2d_magnitude, phase2d_normalize, phase2d_coherence,
    PhaseBank, Coupler, Backbone, Memory, Objective, Sampler, BackboneState,
    Registry, get_registry,
    V4Config, load_config, get_default_config,
)
from .model import QuantumPhaseFieldLLM, create_model
from .banks import SemanticPhaseBank, ContextPhaseBank, LanguagePhaseBank
from .backbone import OscillatorySSM
from .coupler import InterferenceCoupler
from .memory import PhaseAssociativeMemory
from .objectives import CrossEntropyObjective, CoherenceObjective, EnergyObjective
from .sampler import AutoregressiveSampler

__version__ = "0.1.0"

__all__ = [
    # Main model
    'QuantumPhaseFieldLLM', 'create_model',
    # Core
    'Phase2D', 'Phase2DEmbed', 'Phase2DLinear', 'Phase2DLayerNorm', 'IotaBlock',
    'PhaseBank', 'Coupler', 'Backbone', 'Memory', 'Objective', 'Sampler',
    'Registry', 'get_registry',
    'V4Config', 'load_config', 'get_default_config',
    # Implementations
    'SemanticPhaseBank', 'ContextPhaseBank', 'LanguagePhaseBank',
    'OscillatorySSM',
    'InterferenceCoupler',
    'PhaseAssociativeMemory',
    'CrossEntropyObjective', 'CoherenceObjective', 'EnergyObjective',
    'AutoregressiveSampler',
]
