#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Memory: Long-term phase-coded associative memory

Stores knowledge as phase-coded key-value pairs.
Supports incremental learning via memory shards.
"""

from .phase_associative import PhaseAssociativeMemory

__all__ = ['PhaseAssociativeMemory']
