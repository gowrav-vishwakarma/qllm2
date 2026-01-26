#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Metrics: Philosophy-aligned metrics for model introspection

Implements measurable signals inspired by Indian philosophical concepts:
- Manas (मनस्): Active mind/sequence state - backbone activity
- Buddhi (बुद्धि): Discriminative intelligence - decision confidence
- Viveka (विवेक): Discernment/stability - coherence and energy stability
- Smriti (स्मृति): Memory recall - attention sharpness and memory usage
"""

from .philosophy_metrics import (
    PhilosophyMetrics,
    compute_manas_metrics,
    compute_buddhi_metrics,
    compute_viveka_metrics,
    compute_smriti_metrics,
    MetricsLogger,
)

__all__ = [
    'PhilosophyMetrics',
    'compute_manas_metrics',
    'compute_buddhi_metrics',
    'compute_viveka_metrics',
    'compute_smriti_metrics',
    'MetricsLogger',
]
