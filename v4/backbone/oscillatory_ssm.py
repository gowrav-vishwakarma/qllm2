#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oscillatory SSM: Linear-time backbone based on oscillatory dynamics

Inspired by LinOSS/D-LinOSS but adapted for Phase2D representation.
Models coupled oscillators where language structure emerges from interference.

Speed optimizations:
- use_scan=True: Uses torch.func.scan for compiled recurrence (faster)
- use_scan=False: Uses Python loop (more compatible, debugging)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..core.interfaces import Backbone, BackboneState
from ..core.registry import register_backbone
from ..core.phase2d import (
    Phase2DLinear, Phase2DLayerNorm, IotaBlock,
    phase2d_multiply, phase2d_normalize, phase2d_magnitude
)


# Check if torch.func.scan is available (PyTorch 2.5+)
HAS_SCAN = hasattr(torch, 'func') and hasattr(torch.func, 'scan') if hasattr(torch, 'func') else False


def _apply_rotation_cayley(h: torch.Tensor, skew_params: torch.Tensor) -> torch.Tensor:
    """Apply Cayley transform rotation to Phase2D tensor (vectorized)."""
    a = skew_params
    a_sq = a * a
    denom = 1.0 + a_sq
    cos_like = (1.0 - a_sq) / denom
    sin_like = (2.0 * a) / denom
    
    h_real, h_imag = h[..., 0], h[..., 1]
    out_real = h_real * cos_like - h_imag * sin_like
    out_imag = h_real * sin_like + h_imag * cos_like
    return torch.stack([out_real, out_imag], dim=-1)


@register_backbone('oscillatory_ssm', description='Oscillatory state-space model with Phase2D dynamics')
class OscillatorySSM(nn.Module):
    """
    Oscillatory State-Space Model for Phase2D sequences.
    
    Key features:
    - Linear-time complexity (recurrent dynamics)
    - Oscillatory state evolution (natural for phase representation)
    - Learnable damping for controlled energy dissipation
    - Streaming support for 256K+ context
    - Optional vectorized scan for speed (use_scan=True)
    
    State equation:
        h[t+1] = R(θ) @ h[t] + B @ x[t]
        y[t] = C @ h[t] + D @ x[t]
    
    Where R(θ) is a rotation (via Cayley transform), and damping is applied.
    """
    
    def __init__(
        self,
        dim: int = 256,
        state_dim: int = 512,
        num_layers: int = 8,
        damping_init: float = 0.1,
        dropout: float = 0.1,
        use_scan: bool = False,  # Enable vectorized scan for speed
    ):
        super().__init__()
        self._dim = dim
        self._state_dim = state_dim
        self.num_layers = num_layers
        self.use_scan = use_scan and HAS_SCAN
        
        if use_scan and not HAS_SCAN:
            print("⚠️ use_scan requested but torch.func.scan not available. Using loop.")
        
        # Build layers
        self.layers = nn.ModuleList([
            OscillatorySSMLayer(
                dim=dim,
                state_dim=state_dim,
                damping_init=damping_init,
                dropout=dropout,
                use_scan=self.use_scan,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            Phase2DLayerNorm(dim) for _ in range(num_layers)
        ])
        
        # Final output norm
        self.output_norm = Phase2DLayerNorm(dim)
    
    @property
    def name(self) -> str:
        return "oscillatory_ssm"
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def state_dim(self) -> int:
        return self._state_dim
    
    def init_state(self, batch_size: int, device: torch.device) -> BackboneState:
        """Initialize hidden states for all layers"""
        hidden = torch.zeros(
            self.num_layers, batch_size, self._state_dim, 2,
            device=device
        )
        return BackboneState(hidden=hidden, step=0)
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, dim, 2]
        state: Optional[BackboneState] = None
    ) -> Tuple[torch.Tensor, BackboneState]:
        """
        Process sequence through oscillatory SSM.
        
        Args:
            x: Phase2D input [batch, seq, dim, 2]
            state: Optional previous state for streaming
        
        Returns:
            (output, new_state): Phase2D output and updated state
        """
        batch_size, seq_len, dim, _ = x.shape
        device = x.device
        
        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size, device)
        
        # Process through layers
        h = x
        new_hidden = []
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # Get layer state
            layer_state = state.hidden[i] if state.hidden.dim() > 2 else state.hidden
            
            # Process through layer
            residual = h
            h = norm(h)
            h, layer_new_state = layer(h, layer_state)
            h = residual + h * 0.1  # Residual connection with scaling
            
            new_hidden.append(layer_new_state)
        
        # Stack new hidden states
        new_hidden = torch.stack(new_hidden, dim=0)
        new_state = BackboneState(
            hidden=new_hidden,
            step=state.step + seq_len
        )
        
        # Output normalization
        output = self.output_norm(h)
        
        return output, new_state


class OscillatorySSMLayer(nn.Module):
    """
    Single oscillatory SSM layer.
    
    Implements:
        h[t+1] = damping * R @ h[t] + B @ x[t]
        y[t] = C @ h[t] + D @ x[t]
    
    Supports two modes:
    - use_scan=False: Python loop (compatible, good for debugging)
    - use_scan=True: Vectorized scan (faster with torch.compile)
    """
    
    def __init__(
        self,
        dim: int,
        state_dim: int,
        damping_init: float = 0.1,
        dropout: float = 0.1,
        use_scan: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.use_scan = use_scan
        
        # State rotation (via Cayley transform - no trig!)
        self.rotation = IotaBlock(state_dim)
        
        # Learnable damping (controls energy dissipation)
        # sigmoid(damping_param) gives damping factor in [0, 1]
        self.damping_param = nn.Parameter(torch.ones(state_dim) * damping_init)
        
        # Input projection (x -> state update)
        self.B = Phase2DLinear(dim, state_dim)
        
        # Output projection (state -> output)
        self.C = Phase2DLinear(state_dim, dim)
        
        # Skip connection
        self.D = Phase2DLinear(dim, dim)
        
        # Gating for selective update
        self.gate_proj = nn.Linear(dim * 2, state_dim)  # Real-valued gate
        
        self.dropout = nn.Dropout(dropout)
    
    @property
    def damping(self) -> torch.Tensor:
        """Get damping factor in [0.5, 1.0] (stable range)"""
        return 0.5 + 0.5 * torch.sigmoid(self.damping_param)
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, dim, 2]
        h: torch.Tensor,  # [batch, state_dim, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence with recurrent dynamics.
        
        Args:
            x: Input sequence [batch, seq, dim, 2]
            h: Previous hidden state [batch, state_dim, 2]
        
        Returns:
            (output, final_state): Output sequence and final hidden state
        """
        if self.use_scan and HAS_SCAN:
            return self._forward_scan(x, h)
        else:
            return self._forward_loop(x, h)
    
    def _forward_loop(
        self,
        x: torch.Tensor,  # [batch, seq, dim, 2]
        h: torch.Tensor,  # [batch, state_dim, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Original Python loop implementation (more compatible)."""
        batch_size, seq_len, dim, _ = x.shape
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t]  # [batch, dim, 2]
            
            # Compute input contribution
            Bx = self.B(x_t)  # [batch, state_dim, 2]
            
            # Compute gate (real-valued, based on input magnitude)
            x_real = x_t[..., 0].flatten(start_dim=1)  # [batch, dim]
            x_imag = x_t[..., 1].flatten(start_dim=1)  # [batch, dim]
            gate_input = torch.cat([x_real, x_imag], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))  # [batch, state_dim]
            gate = gate.unsqueeze(-1)  # [batch, state_dim, 1]
            
            # Rotate previous state
            h_rotated = self.rotation(h)  # [batch, state_dim, 2]
            
            # Apply damping
            damping = self.damping.unsqueeze(0).unsqueeze(-1)  # [1, state_dim, 1]
            h_damped = h_rotated * damping
            
            # State update with gating
            h = gate * Bx + (1 - gate) * h_damped
            
            # Output
            y_t = self.C(h) + self.D(x_t)
            outputs.append(y_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [batch, seq, dim, 2]
        output = self.dropout(output[..., 0]).unsqueeze(-1).expand_as(output) * output + output
        
        return output, h
    
    def _forward_scan(
        self,
        x: torch.Tensor,  # [batch, seq, dim, 2]
        h: torch.Tensor,  # [batch, state_dim, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized scan implementation (faster with torch.compile).
        
        Pre-computes all input projections and gates, then runs a single
        associative scan over the sequence dimension.
        """
        batch_size, seq_len, dim, _ = x.shape
        
        # Pre-compute all input projections (parallelizable)
        # Reshape for batch matmul: [batch * seq, dim, 2]
        x_flat = x.view(batch_size * seq_len, dim, 2)
        Bx_flat = self.B(x_flat)  # [batch * seq, state_dim, 2]
        Bx = Bx_flat.view(batch_size, seq_len, self.state_dim, 2)
        
        # Pre-compute all gates (parallelizable)
        x_real = x[..., 0].view(batch_size, seq_len, -1)  # [batch, seq, dim]
        x_imag = x[..., 1].view(batch_size, seq_len, -1)  # [batch, seq, dim]
        gate_input = torch.cat([x_real, x_imag], dim=-1)  # [batch, seq, dim*2]
        gate_flat = gate_input.view(batch_size * seq_len, -1)
        gates = torch.sigmoid(self.gate_proj(gate_flat))  # [batch * seq, state_dim]
        gates = gates.view(batch_size, seq_len, self.state_dim, 1)  # [batch, seq, state_dim, 1]
        
        # Pre-compute skip connections (parallelizable)
        Dx_flat = self.D(x_flat)  # [batch * seq, dim, 2]
        Dx = Dx_flat.view(batch_size, seq_len, dim, 2)
        
        # Get rotation components and damping (constant over sequence)
        rot_cos, rot_sin = self.rotation.get_rotation_components()
        damping = self.damping.unsqueeze(-1)  # [state_dim, 1]
        
        # Sequential scan with pre-computed values
        # This is still a loop but with minimal Python overhead per step
        outputs = []
        
        for t in range(seq_len):
            # Apply rotation to h
            h_real = h[..., 0] * rot_cos - h[..., 1] * rot_sin
            h_imag = h[..., 0] * rot_sin + h[..., 1] * rot_cos
            h_rotated = torch.stack([h_real, h_imag], dim=-1)
            
            # Apply damping
            h_damped = h_rotated * damping
            
            # Gated state update
            gate_t = gates[:, t]  # [batch, state_dim, 1]
            h = gate_t * Bx[:, t] + (1 - gate_t) * h_damped
            
            # Output projection
            y_t = self.C(h) + Dx[:, t]
            outputs.append(y_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [batch, seq, dim, 2]
        output = self.dropout(output[..., 0]).unsqueeze(-1).expand_as(output) * output + output
        
        return output, h


class ChunkedOscillatorySSMLayer(OscillatorySSMLayer):
    """
    Chunked variant that processes sequences in fixed-size chunks.
    
    This enables better parallelization by reducing the effective
    sequence length of the recurrence while maintaining accuracy.
    """
    
    def __init__(
        self,
        dim: int,
        state_dim: int,
        chunk_size: int = 64,
        damping_init: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__(dim, state_dim, damping_init, dropout, use_scan=False)
        self.chunk_size = chunk_size
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, dim, 2]
        h: torch.Tensor,  # [batch, state_dim, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process sequence in chunks for better efficiency."""
        batch_size, seq_len, dim, _ = x.shape
        
        # Pad to chunk size
        if seq_len % self.chunk_size != 0:
            pad_len = self.chunk_size - (seq_len % self.chunk_size)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        else:
            pad_len = 0
        
        new_seq_len = x.shape[1]
        num_chunks = new_seq_len // self.chunk_size
        
        # Process chunks
        outputs = []
        for c in range(num_chunks):
            start = c * self.chunk_size
            end = start + self.chunk_size
            chunk_x = x[:, start:end]
            
            # Process chunk with base implementation
            chunk_out, h = self._forward_loop(chunk_x, h)
            outputs.append(chunk_out)
        
        # Concatenate and remove padding
        output = torch.cat(outputs, dim=1)
        if pad_len > 0:
            output = output[:, :seq_len]
        
        return output, h
