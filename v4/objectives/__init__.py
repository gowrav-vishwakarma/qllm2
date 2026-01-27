#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Objectives: Loss functions for training

Multiple objectives can be combined with weights.
"""

from .ce import CrossEntropyObjective
from .coherence import CoherenceObjective, EnergyObjective, CouplingObjective

__all__ = ['CrossEntropyObjective', 'CoherenceObjective', 'EnergyObjective', 'CouplingObjective']
