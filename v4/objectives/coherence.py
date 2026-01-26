#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coherence and Energy Objectives: Phase-space regularization

GPU-friendly coherence/energy losses (no trig in hot path).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from ..core.interfaces import Objective, ObjectiveResult
from ..core.registry import register_objective
from ..core.phase2d import phase2d_magnitude, phase2d_coherence


@register_objective('coherence', description='Phase coherence regularization (encourages smooth phase transitions)')
class CoherenceObjective(nn.Module):
    """
    Coherence Objective: encourages smooth phase transitions.
    
    Penalizes abrupt phase changes between adjacent tokens.
    Uses normalized dot product (no trig).
    """
    
    def __init__(
        self,
        weight: float = 0.01,
        window_size: int = 4,
    ):
        super().__init__()
        self._weight = weight
        self.window_size = window_size
    
    @property
    def name(self) -> str:
        return "coherence"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        context: Optional[Dict[str, Any]] = None
    ) -> ObjectiveResult:
        """
        Compute coherence loss.
        
        Args:
            model_output: Must contain 'phase_states' [batch, seq, dim, 2]
        """
        if 'phase_states' not in model_output:
            return ObjectiveResult(
                loss=torch.tensor(0.0),
                metrics={'coherence_loss': 0.0, 'avg_coherence': 0.0}
            )
        
        phase_states = model_output['phase_states']  # [batch, seq, dim, 2]
        batch_size, seq_len, dim, _ = phase_states.shape
        
        if seq_len < 2:
            return ObjectiveResult(
                loss=torch.tensor(0.0, device=phase_states.device),
                metrics={'coherence_loss': 0.0, 'avg_coherence': 1.0}
            )
        
        # Compute local coherence (adjacent tokens)
        # coherence(t, t+1) via normalized dot product
        current = phase_states[:, :-1]  # [batch, seq-1, dim, 2]
        next_t = phase_states[:, 1:]    # [batch, seq-1, dim, 2]
        
        # Coherence = Re(current * conj(next)) / (|current| * |next|)
        curr_real, curr_imag = current[..., 0], current[..., 1]
        next_real, next_imag = next_t[..., 0], next_t[..., 1]
        
        # Real part of complex dot product
        dot_real = (curr_real * next_real + curr_imag * next_imag).sum(dim=-1)
        
        # Magnitudes
        curr_mag = torch.sqrt((curr_real ** 2 + curr_imag ** 2).sum(dim=-1) + 1e-8)
        next_mag = torch.sqrt((next_real ** 2 + next_imag ** 2).sum(dim=-1) + 1e-8)
        
        # Normalized coherence (in [-1, 1], higher = more coherent)
        coherence = dot_real / (curr_mag * next_mag + 1e-8)
        
        # Loss: encourage high coherence (penalize low coherence)
        # We want coherence close to 1, so loss = 1 - coherence
        coherence_loss = (1 - coherence).mean()
        
        avg_coherence = coherence.mean().item()
        
        return ObjectiveResult(
            loss=coherence_loss,
            metrics={
                'coherence_loss': coherence_loss.item(),
                'avg_coherence': avg_coherence,
            }
        )


@register_objective('energy', description='Phase energy regularization (encourages stable magnitudes)')
class EnergyObjective(nn.Module):
    """
    Energy Objective: encourages stable phase magnitudes.
    
    Penalizes phase states with very high or very low energy (magnitude).
    Helps prevent gradient explosion/vanishing in phase space.
    """
    
    def __init__(
        self,
        weight: float = 0.001,
        target_magnitude: float = 1.0,
    ):
        super().__init__()
        self._weight = weight
        self.target_magnitude = target_magnitude
    
    @property
    def name(self) -> str:
        return "energy"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        context: Optional[Dict[str, Any]] = None
    ) -> ObjectiveResult:
        """
        Compute energy loss.
        
        Args:
            model_output: Must contain 'phase_states' [batch, seq, dim, 2]
        """
        if 'phase_states' not in model_output:
            return ObjectiveResult(
                loss=torch.tensor(0.0),
                metrics={'energy_loss': 0.0, 'avg_magnitude': 0.0}
            )
        
        phase_states = model_output['phase_states']
        
        # Compute magnitude
        magnitude = phase2d_magnitude(phase_states)  # [batch, seq, dim]
        
        # Energy loss: penalize deviation from target magnitude
        energy_loss = ((magnitude - self.target_magnitude) ** 2).mean()
        
        avg_magnitude = magnitude.mean().item()
        
        return ObjectiveResult(
            loss=energy_loss,
            metrics={
                'energy_loss': energy_loss.item(),
                'avg_magnitude': avg_magnitude,
            }
        )


@register_objective('coupling', description='Cross-bank coupling loss (from coupler)')
class CouplingObjective(nn.Module):
    """
    Coupling Objective: uses the coupler's cross-bank coherence loss.
    
    This is a pass-through that extracts the coupling loss from context.
    """
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self._weight = weight
    
    @property
    def name(self) -> str:
        return "coupling"
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        context: Optional[Dict[str, Any]] = None
    ) -> ObjectiveResult:
        """
        Extract coupling loss from context.
        """
        if context is None or 'coupling_loss' not in context:
            return ObjectiveResult(
                loss=torch.tensor(0.0),
                metrics={'coupling_loss': 0.0}
            )
        
        coupling_loss = context['coupling_loss']
        
        return ObjectiveResult(
            loss=coupling_loss,
            metrics={'coupling_loss': coupling_loss.item()}
        )
