#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Associative Memory: Long-term storage with phase-coded retrieval

Uses phase coherence for soft attention over memory slots.
Supports incremental learning through memory shards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..core.interfaces import Memory, MemoryReadResult
from ..core.registry import register_memory
from ..core.phase2d import (
    Phase2DLinear, Phase2DLayerNorm,
    phase2d_multiply, phase2d_normalize, phase2d_magnitude, phase2d_conjugate
)


@register_memory('phase_associative', description='Phase-coded associative memory with coherence-based retrieval')
class PhaseAssociativeMemory(nn.Module):
    """
    Phase Associative Memory: stores and retrieves via phase coherence.
    
    Key features:
    - Phase-coded key-value storage
    - Coherence-based soft retrieval (no trig)
    - Memory consolidation for important patterns
    - Shard support for incremental learning
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_slots: int = 4096,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._dim = dim
        self._num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Memory slots: keys and values (phase-coded)
        self.keys = nn.Parameter(torch.randn(num_slots, dim, 2) * 0.02)
        self.values = nn.Parameter(torch.randn(num_slots, dim, 2) * 0.02)
        
        # Slot importance (for consolidation)
        self.slot_importance = nn.Parameter(torch.zeros(num_slots))
        
        # Query projection
        self.query_proj = Phase2DLinear(dim, dim)
        
        # Output projection
        self.output_proj = Phase2DLinear(dim, dim)
        self.output_norm = Phase2DLayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Memory shards for incremental learning
        self.shards: Dict[str, Dict[str, torch.Tensor]] = {}
    
    @property
    def name(self) -> str:
        return "phase_associative"
    
    @property
    def num_slots(self) -> int:
        return self._num_slots
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def _compute_attention(
        self,
        query: torch.Tensor,  # [batch, seq, dim, 2]
        keys: torch.Tensor,   # [num_slots, dim, 2]
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute attention weights via phase coherence.
        
        Uses memory-efficient chunked computation to avoid OOM.
        Uses dot product (real part) as coherence measure - no trig!
        """
        batch_size, seq_len, dim, _ = query.shape
        num_slots = keys.shape[0]
        
        # Memory-efficient: reshape query to [batch*seq, dim, 2]
        query_flat = query.view(batch_size * seq_len, dim, 2)
        
        # Pre-compute key magnitudes [num_slots]
        k_real = keys[..., 0]  # [num_slots, dim]
        k_imag = keys[..., 1]  # [num_slots, dim]
        k_mag = torch.sqrt((k_real ** 2 + k_imag ** 2).sum(dim=-1) + 1e-8)  # [num_slots]
        
        # Pre-compute query magnitudes [batch*seq]
        q_real = query_flat[..., 0]  # [batch*seq, dim]
        q_imag = query_flat[..., 1]  # [batch*seq, dim]
        q_mag = torch.sqrt((q_real ** 2 + q_imag ** 2).sum(dim=-1) + 1e-8)  # [batch*seq]
        
        # Compute coherence efficiently using einsum: [batch*seq, num_slots]
        # Re(q * conj(k)) = q_r * k_r + q_i * k_i, summed over dim
        coherence = torch.einsum('bd,nd->bn', q_real, k_real) + torch.einsum('bd,nd->bn', q_imag, k_imag)
        
        # Normalize by magnitudes
        coherence = coherence / (q_mag.unsqueeze(-1) * k_mag.unsqueeze(0) + 1e-8)
        
        # Apply top-k if specified
        if top_k is not None and top_k < num_slots:
            # Keep only top-k values, set rest to -inf
            topk_vals, topk_idx = coherence.topk(top_k, dim=-1)
            mask = torch.zeros_like(coherence)
            mask.scatter_(-1, topk_idx, 1.0)
            coherence = coherence * mask + (1 - mask) * (-1e9)
        
        # Softmax to get attention weights
        attention = F.softmax(coherence, dim=-1)  # [batch*seq, num_slots]
        
        # Reshape back to [batch, seq, num_slots]
        attention = attention.view(batch_size, seq_len, num_slots)
        
        return attention
    
    def read(
        self,
        query: torch.Tensor,
        top_k: Optional[int] = None
    ) -> MemoryReadResult:
        """
        Read from memory using phase-coded query.
        """
        batch_size, seq_len, dim, _ = query.shape
        
        # Project query
        query_proj = self.query_proj(query)
        
        # Compute attention over main memory
        attention = self._compute_attention(query_proj, self.keys, top_k)
        
        # Retrieve values
        # [batch, seq, num_slots] @ [num_slots, dim, 2] -> [batch, seq, dim, 2]
        retrieved = torch.einsum('bsn,ndp->bsdp', attention, self.values)
        
        # Also query shards if present
        for shard_id, shard in self.shards.items():
            shard_attn = self._compute_attention(query_proj, shard['keys'], top_k)
            shard_vals = torch.einsum('bsn,ndp->bsdp', shard_attn, shard['values'])
            retrieved = retrieved + shard_vals * 0.5  # Lower weight for shards
        
        # Output projection
        output = self.output_norm(retrieved)
        output = self.output_proj(output)
        
        return MemoryReadResult(
            values=output,
            attention=attention,
        )
    
    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> None:
        """
        Write to memory (find least important slot and overwrite).
        """
        if not self.training:
            return
        
        # Average over batch/seq if present
        if key.dim() > 3:
            key = key.mean(dim=(0, 1))  # [dim, 2]
        elif key.dim() > 2:
            key = key.mean(dim=0)  # [dim, 2]
        
        if value.dim() > 3:
            value = value.mean(dim=(0, 1))
        elif value.dim() > 2:
            value = value.mean(dim=0)
        
        # Find least important slot
        least_important_idx = self.slot_importance.argmin()
        
        # Write with momentum (soft update)
        momentum = 0.9
        self.keys.data[least_important_idx] = (
            momentum * self.keys.data[least_important_idx] +
            (1 - momentum) * key
        )
        self.values.data[least_important_idx] = (
            momentum * self.values.data[least_important_idx] +
            (1 - momentum) * value
        )
        
        # Update importance
        if importance is not None:
            self.slot_importance.data[least_important_idx] = importance.mean().item()
        else:
            self.slot_importance.data[least_important_idx] = 1.0
    
    def consolidate(self) -> None:
        """Decay importance of unused slots."""
        self.slot_importance.data *= 0.99
    
    def get_shard(self, shard_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get a memory shard for saving."""
        if shard_id in self.shards:
            return self.shards[shard_id]
        return None
    
    def add_shard(self, shard_id: str, shard: Dict[str, torch.Tensor]) -> None:
        """Add a memory shard (for incremental learning)."""
        self.shards[shard_id] = shard
    
    def create_shard_from_current(self, shard_id: str, top_k: int = 256) -> None:
        """
        Create a shard from the most important current slots.
        Useful for saving learned knowledge before fine-tuning.
        """
        # Get top-k most important slots
        _, topk_idx = self.slot_importance.topk(top_k)
        
        shard = {
            'keys': self.keys.data[topk_idx].clone(),
            'values': self.values.data[topk_idx].clone(),
            'importance': self.slot_importance.data[topk_idx].clone(),
        }
        
        self.shards[shard_id] = shard
