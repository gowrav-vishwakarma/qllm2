#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Quantum-Inspired LLM
"""

import os
import json
import torch
from typing import Dict, Any, Optional

def device_str() -> str:
    """Get the best available device string"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def save_checkpoint(state: Dict[str, Any], path: str):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, device: str = 'cpu'):
    """Load model checkpoint"""
    return torch.load(path, map_location=device)

def save_args(args: Dict[str, Any], path: str):
    """Save training arguments"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(args, f, indent=2)

def load_args(path: str) -> Dict[str, Any]:
    """Load training arguments"""
    with open(path, 'r') as f:
        return json.load(f)

def get_model_size(model: torch.nn.Module) -> int:
    """Get model size in parameters"""
    return sum(p.numel() for p in model.parameters())

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in GB"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'cached': torch.cuda.memory_reserved() / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated() / 1e9
        }
    return {'allocated': 0, 'cached': 0, 'max_allocated': 0}

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_size(bytes: int) -> str:
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}TB"