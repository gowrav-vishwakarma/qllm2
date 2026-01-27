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
        top_k: Optional[int] = None,
        return_sparse: bool = False,
    ) -> torch.Tensor:
        """
        Compute attention weights via phase coherence.
        
        Uses memory-efficient chunked top-k computation to avoid OOM.
        Never materializes the full [batch*seq, num_slots] matrix.
        Uses dot product (real part) as coherence measure - no trig!
        
        Args:
            query: [batch, seq, dim, 2] query vectors
            keys: [num_slots, dim, 2] memory keys
            top_k: Number of top keys to keep (default: 64)
            return_sparse: If True, return sparse attention (indices + values)
        
        Returns:
            attention: [batch, seq, num_slots] attention weights (sparse if return_sparse)
        """
        batch_size, seq_len, dim, _ = query.shape
        num_slots = keys.shape[0]
        
        # Default top_k
        if top_k is None:
            top_k = min(64, num_slots)
        
        # Memory-efficient: reshape query to [batch*seq, dim, 2]
        num_queries = batch_size * seq_len
        query_flat = query.view(num_queries, dim, 2)
        
        # Pre-compute query components
        q_real = query_flat[..., 0]  # [num_queries, dim]
        q_imag = query_flat[..., 1]  # [num_queries, dim]
        q_mag = torch.sqrt((q_real ** 2 + q_imag ** 2).sum(dim=-1) + 1e-8)  # [num_queries]
        
        # Chunk size for processing keys (tuned for memory efficiency)
        chunk_size = min(2048, num_slots)
        
        # Initialize running top-k trackers
        # topk_values: [num_queries, top_k] - best coherence scores so far
        # topk_indices: [num_queries, top_k] - corresponding slot indices
        device = query.device
        topk_values = torch.full((num_queries, top_k), float('-inf'), device=device)
        topk_indices = torch.zeros((num_queries, top_k), dtype=torch.long, device=device)
        
        # Process keys in chunks
        for chunk_start in range(0, num_slots, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_slots)
            chunk_keys = keys[chunk_start:chunk_end]  # [chunk_size, dim, 2]
            chunk_size_actual = chunk_end - chunk_start
            
            # Pre-compute chunk key components
            k_real = chunk_keys[..., 0]  # [chunk_size, dim]
            k_imag = chunk_keys[..., 1]  # [chunk_size, dim]
            k_mag = torch.sqrt((k_real ** 2 + k_imag ** 2).sum(dim=-1) + 1e-8)  # [chunk_size]
            
            # Compute coherence for this chunk: [num_queries, chunk_size]
            # Re(q * conj(k)) = q_r * k_r + q_i * k_i, summed over dim
            chunk_coherence = (
                torch.einsum('qd,cd->qc', q_real, k_real) +
                torch.einsum('qd,cd->qc', q_imag, k_imag)
            )
            
            # Normalize by magnitudes
            chunk_coherence = chunk_coherence / (q_mag.unsqueeze(-1) * k_mag.unsqueeze(0) + 1e-8)
            
            # Create indices for this chunk (offset by chunk_start)
            chunk_indices = torch.arange(chunk_start, chunk_end, device=device)
            chunk_indices = chunk_indices.unsqueeze(0).expand(num_queries, -1)  # [num_queries, chunk_size]
            
            # Merge with running top-k
            # Concatenate current top-k with chunk results
            combined_values = torch.cat([topk_values, chunk_coherence], dim=-1)  # [num_queries, top_k + chunk_size]
            combined_indices = torch.cat([topk_indices, chunk_indices], dim=-1)  # [num_queries, top_k + chunk_size]
            
            # Take new top-k
            new_topk_values, new_topk_local_idx = combined_values.topk(top_k, dim=-1)
            
            # Gather corresponding global indices
            new_topk_indices = torch.gather(combined_indices, dim=-1, index=new_topk_local_idx)
            
            topk_values = new_topk_values
            topk_indices = new_topk_indices
        
        # Compute softmax over top-k scores
        attention_sparse = F.softmax(topk_values, dim=-1)  # [num_queries, top_k]
        
        if return_sparse:
            # Return sparse representation (useful for very large memory)
            return attention_sparse.view(batch_size, seq_len, top_k), topk_indices.view(batch_size, seq_len, top_k)
        
        # Scatter sparse attention into dense tensor
        # This is the memory-expensive part, but now bounded by batch*seq*num_slots
        # For very large num_slots, consider always using return_sparse=True
        attention = torch.zeros(num_queries, num_slots, device=device)
        attention.scatter_(-1, topk_indices, attention_sparse)
        
        # Reshape back to [batch, seq, num_slots]
        attention = attention.view(batch_size, seq_len, num_slots)
        
        return attention
    
    def _compute_attention_sparse(
        self,
        query: torch.Tensor,  # [batch, seq, dim, 2]
        keys: torch.Tensor,   # [num_slots, dim, 2]
        values: torch.Tensor,  # [num_slots, dim, 2]
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Memory-efficient retrieval that never creates dense [batch*seq, num_slots].
        
        Uses chunked top-k and direct value retrieval.
        
        Returns:
            retrieved: [batch, seq, dim, 2] retrieved values
        """
        batch_size, seq_len, dim, _ = query.shape
        num_slots = keys.shape[0]
        
        # Default top_k
        if top_k is None:
            top_k = min(64, num_slots)
        
        # Get sparse attention: [batch, seq, top_k] weights and indices
        attn_weights, attn_indices = self._compute_attention(
            query, keys, top_k=top_k, return_sparse=True
        )
        
        # Retrieve values using gathered indices
        # attn_indices: [batch, seq, top_k] - indices into values
        # values: [num_slots, dim, 2]
        
        # Flatten for gathering
        num_queries = batch_size * seq_len
        attn_indices_flat = attn_indices.view(num_queries, top_k)  # [num_queries, top_k]
        attn_weights_flat = attn_weights.view(num_queries, top_k)  # [num_queries, top_k]
        
        # Gather values: [top_k, dim, 2] for each query
        # Use advanced indexing: values[attn_indices] gives [num_queries, top_k, dim, 2]
        gathered_values = values[attn_indices_flat]  # [num_queries, top_k, dim, 2]
        
        # Weighted sum over top_k dimension
        # attn_weights_flat: [num_queries, top_k] -> [num_queries, top_k, 1, 1]
        weighted_values = gathered_values * attn_weights_flat.unsqueeze(-1).unsqueeze(-1)
        retrieved = weighted_values.sum(dim=1)  # [num_queries, dim, 2]
        
        # Reshape back
        retrieved = retrieved.view(batch_size, seq_len, dim, 2)
        
        return retrieved
    
    def read(
        self,
        query: torch.Tensor,
        top_k: Optional[int] = None,
        use_sparse: bool = True,
    ) -> MemoryReadResult:
        """
        Read from memory using phase-coded query.
        
        Args:
            query: [batch, seq, dim, 2] query vectors
            top_k: Number of memory slots to retrieve (default: 64)
            use_sparse: If True, use memory-efficient sparse retrieval
        
        Returns:
            MemoryReadResult with retrieved values and attention weights
        """
        batch_size, seq_len, dim, _ = query.shape
        
        # Default top_k for efficiency
        if top_k is None:
            top_k = min(64, self._num_slots)
        
        # Project query
        query_proj = self.query_proj(query)
        
        if use_sparse:
            # Memory-efficient sparse retrieval (never creates [batch*seq, num_slots])
            retrieved = self._compute_attention_sparse(
                query_proj, self.keys, self.values, top_k
            )
            # For sparse mode, we only return sparse attention for debugging
            # Set attention to None to indicate sparse mode was used
            attention = None
        else:
            # Original dense path (for debugging/metrics)
            attention = self._compute_attention(query_proj, self.keys, top_k)
            retrieved = torch.einsum('bsn,ndp->bsdp', attention, self.values)
        
        # Also query shards if present
        for shard_id, shard in self.shards.items():
            if use_sparse:
                shard_vals = self._compute_attention_sparse(
                    query_proj, shard['keys'], shard['values'], top_k
                )
            else:
                shard_attn = self._compute_attention(query_proj, shard['keys'], top_k)
                shard_vals = torch.einsum('bsn,ndp->bsdp', shard_attn, shard['values'])
            retrieved = retrieved + shard_vals * 0.5  # Lower weight for shards
        
        # Output projection
        output = self.output_norm(retrieved)
        output = self.output_proj(output)
        
        return MemoryReadResult(
            values=output,
            attention=attention,  # None if sparse mode
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
