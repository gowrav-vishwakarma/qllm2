#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Phase Banks: Implementations of different phase-space layers

Each bank represents a separate "layer" of meaning (semantic, context, language, emotion).
Banks are combined by the Coupler via interference.
"""

from .semantic import SemanticPhaseBank
from .context import ContextPhaseBank
from .language import LanguagePhaseBank

__all__ = ['SemanticPhaseBank', 'ContextPhaseBank', 'LanguagePhaseBank']
