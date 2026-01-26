#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Backbones: Linear-time sequence processing engines

The backbone handles sequence-level processing with linear complexity.
Supports streaming for 256K+ context.

Speed options:
- use_scan=True: Enable vectorized scan (faster with torch.compile)
- ChunkedOscillatorySSMLayer: Process in fixed-size chunks (better parallelization)
"""

from .oscillatory_ssm import OscillatorySSM, OscillatorySSMLayer, ChunkedOscillatorySSMLayer

__all__ = ['OscillatorySSM', 'OscillatorySSMLayer', 'ChunkedOscillatorySSMLayer']
