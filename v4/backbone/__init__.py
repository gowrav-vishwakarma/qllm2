#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Backbones: Linear-time sequence processing engines

The backbone handles sequence-level processing with linear complexity.
Supports streaming for 256K+ context.
"""

from .oscillatory_ssm import OscillatorySSM

__all__ = ['OscillatorySSM']
