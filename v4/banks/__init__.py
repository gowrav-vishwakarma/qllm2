#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Phase Banks: Implementations of different phase-space layers

Each bank represents a separate "layer" of meaning:
- Semantic: Core word meanings and concepts
- Context: Local compositional structure
- Language: Language/dialect-specific patterns
- Morphology: Grammatical transformations (user idea: root+affix)
- Orthography: Script/shape patterns for multilingual support

Banks are combined by the Coupler via interference.
"""

from .semantic import SemanticPhaseBank
from .context import ContextPhaseBank
from .language import LanguagePhaseBank
from .morphology import MorphologyPhaseBank
from .orthography import OrthographyPhaseBank

__all__ = [
    'SemanticPhaseBank',
    'ContextPhaseBank',
    'LanguagePhaseBank',
    'MorphologyPhaseBank',
    'OrthographyPhaseBank',
]
