#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Couplers: Combine multiple phase banks via interference

The coupler takes outputs from all phase banks and combines them
into a unified representation using GPU-friendly interference operations.
"""

from .interference import InterferenceCoupler

__all__ = ['InterferenceCoupler']
