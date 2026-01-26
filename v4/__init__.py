#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Quantum Phase-Field LLM

A novel architecture combining:
- Phase2D representation (complex/phase without trig)
- Multi-layer phase banks (semantic/context/language/morphology/orthography)
- Oscillatory SSM backbone (linear-time)
- Interference coupling
- Phase-coded associative memory
- Incremental learning support
- Morphological tokenization (root + affix phase rotations)
- Philosophy-aligned metrics (Manas/Buddhi/Viveka/Smriti)

All operations are GPU-friendly (GEMM-based, Tensor Core optimized).
"""

from .core import (
    Phase2D, Phase2DEmbed, Phase2DLinear, Phase2DLayerNorm, IotaBlock,
    phase2d_multiply, phase2d_magnitude, phase2d_normalize, phase2d_coherence,
    PhaseBank, Coupler, Backbone, Memory, Objective, Sampler, BackboneState,
    Registry, get_registry,
    V4Config, load_config, get_default_config,
    MorphologyAwareEmbed, AffixRotationBank, DualEmbedding,
)
from .model import QuantumPhaseFieldLLM, create_model
from .banks import (
    SemanticPhaseBank, ContextPhaseBank, LanguagePhaseBank,
    MorphologyPhaseBank, OrthographyPhaseBank,
)
from .backbone import OscillatorySSM
from .coupler import InterferenceCoupler
from .memory import PhaseAssociativeMemory
from .objectives import CrossEntropyObjective, CoherenceObjective, EnergyObjective
from .sampler import AutoregressiveSampler
from .metrics import (
    PhilosophyMetrics, MetricsLogger,
    compute_manas_metrics, compute_buddhi_metrics,
    compute_viveka_metrics, compute_smriti_metrics,
)

__version__ = "0.1.0"

__all__ = [
    # Main model
    'QuantumPhaseFieldLLM', 'create_model',
    # Core
    'Phase2D', 'Phase2DEmbed', 'Phase2DLinear', 'Phase2DLayerNorm', 'IotaBlock',
    'PhaseBank', 'Coupler', 'Backbone', 'Memory', 'Objective', 'Sampler',
    'Registry', 'get_registry',
    'V4Config', 'load_config', 'get_default_config',
    # Morphology embedding
    'MorphologyAwareEmbed', 'AffixRotationBank', 'DualEmbedding',
    # Phase Banks
    'SemanticPhaseBank', 'ContextPhaseBank', 'LanguagePhaseBank',
    'MorphologyPhaseBank', 'OrthographyPhaseBank',
    # Backbone
    'OscillatorySSM',
    # Coupler
    'InterferenceCoupler',
    # Memory
    'PhaseAssociativeMemory',
    # Objectives
    'CrossEntropyObjective', 'CoherenceObjective', 'EnergyObjective',
    # Sampler
    'AutoregressiveSampler',
    # Metrics
    'PhilosophyMetrics', 'MetricsLogger',
    'compute_manas_metrics', 'compute_buddhi_metrics',
    'compute_viveka_metrics', 'compute_smriti_metrics',
]
