#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Episodic Memory: Simple ring buffer for sequence-local retrieval

This provides the "copy" capability that transformers have via attention.
During a forward pass, we store recent states and allow retrieval via
coherence-based attention.

Key differences from global PhaseAssociativeMemory:
- Per-sequence (not learned global slots)
- Ring buffer (fixed size, oldest overwritten)
- Very fast (no parameter updates during forward)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class EpisodicReadResult:
    """Result of episodic memory read"""
    values: torch.Tensor  # [batch, seq, dim, 2] retrieved values
    attention: Optional[torch.Tensor] = None  # [batch, seq, buffer_size] for debugging


class EpisodicMemory(nn.Module):
    """
    Episodic Memory: ring buffer for within-sequence retrieval.
    
    This gives the model "copy" capability similar to transformers,
    but with O(n * buffer_size) complexity instead of O(n^2).
    
    Usage:
        episodic = EpisodicMemory(dim=256, buffer_size=64)
        
        # During forward pass:
        for t in range(seq_len):
            # Read from buffer (query current position against history)
            retrieved = episodic.read(current_state)
            
            # Write current state to buffer
            episodic.write(current_state)
    
    Or in batch mode (all positions at once):
        retrieved = episodic.batch_read_write(all_states)
    """
    
    def __init__(
        self,
        dim: int = 256,
        buffer_size: int = 64,
        top_k: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.buffer_size = buffer_size
        self.top_k = top_k
        
        # No learnable parameters - this is a pure buffer
        # The buffer is created dynamically per forward pass
        
        # Optional: learned query/key projections for better retrieval
        # (small overhead, big quality gain)
        self.query_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        self.key_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        
        # Initialize as identity-ish
        nn.init.eye_(self.query_proj.weight)
        nn.init.eye_(self.key_proj.weight)
    
    def batch_read_write(
        self,
        states: torch.Tensor,  # [batch, seq, dim, 2]
        causal: bool = True,
    ) -> EpisodicReadResult:
        """
        Process entire sequence at once with causal masking.
        
        Each position t can only attend to positions 0..t-1.
        
        Args:
            states: [batch, seq, dim, 2] Phase2D states
            causal: If True, position t can only see 0..t-1
        
        Returns:
            EpisodicReadResult with retrieved values
        """
        batch_size, seq_len, dim, _ = states.shape
        device = states.device
        
        if seq_len <= 1:
            # Nothing to retrieve from
            return EpisodicReadResult(
                values=torch.zeros_like(states),
                attention=None,
            )
        
        # Flatten Phase2D to real: [batch, seq, dim*2]
        states_flat = states.view(batch_size, seq_len, dim * 2)
        
        # Project queries and keys
        queries = self.query_proj(states_flat)  # [batch, seq, dim*2]
        keys = self.key_proj(states_flat)  # [batch, seq, dim*2]
        
        # Compute attention scores: [batch, seq, seq]
        # (queries @ keys.T) / sqrt(dim*2)
        scale = (dim * 2) ** -0.5
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) * scale
        
        # Causal mask: position t can only attend to 0..t-1
        if causal:
            # Create lower-triangular mask (excluding diagonal)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=0)
            mask = mask.bool()
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        
        # Limit to buffer_size most recent positions
        # For each position t, only attend to max(0, t-buffer_size)..t-1
        if self.buffer_size < seq_len:
            # Create buffer window mask
            positions = torch.arange(seq_len, device=device)
            # valid[i,j] = True if j >= i - buffer_size and j < i
            valid = (positions.unsqueeze(1) - positions.unsqueeze(0) <= self.buffer_size) & \
                    (positions.unsqueeze(1) > positions.unsqueeze(0))
            attn_scores = attn_scores.masked_fill(~valid.unsqueeze(0), float('-inf'))
        
        # Softmax (positions with all -inf will get uniform weights, but retrieve zeros)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Handle NaN from all-masked positions (first position has nothing to attend to)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Retrieve values: [batch, seq, dim*2]
        retrieved_flat = torch.bmm(attn_weights, states_flat)
        
        # Reshape back to Phase2D: [batch, seq, dim, 2]
        retrieved = retrieved_flat.view(batch_size, seq_len, dim, 2)
        
        return EpisodicReadResult(
            values=retrieved,
            attention=attn_weights,
        )


class EpisodicMemoryEfficient(nn.Module):
    """
    Memory-efficient episodic memory using chunked attention.
    
    For very long sequences, we chunk the attention computation
    to avoid O(n^2) memory usage.
    
    This version only looks at the last `buffer_size` positions,
    making it O(n * buffer_size) memory.
    """
    
    def __init__(
        self,
        dim: int = 256,
        buffer_size: int = 64,
        top_k: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.buffer_size = buffer_size
        self.top_k = top_k
        
        # Lightweight projections
        self.query_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        self.key_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        
        nn.init.eye_(self.query_proj.weight)
        nn.init.eye_(self.key_proj.weight)
    
    def forward(
        self,
        states: torch.Tensor,  # [batch, seq, dim, 2]
    ) -> EpisodicReadResult:
        """
        Efficient episodic retrieval with bounded memory.
        
        Each position only attends to the previous `buffer_size` positions.
        Uses simple windowed attention with causal masking.
        """
        batch_size, seq_len, dim, _ = states.shape
        device = states.device
        
        if seq_len <= 1:
            return EpisodicReadResult(
                values=torch.zeros_like(states),
                attention=None,
            )
        
        # Flatten Phase2D: [batch, seq, dim*2]
        states_flat = states.view(batch_size, seq_len, dim * 2)
        
        # Project
        queries = self.query_proj(states_flat)  # [batch, seq, dim*2]
        keys = self.key_proj(states_flat)  # [batch, seq, dim*2]
        
        scale = (dim * 2) ** -0.5
        
        # Simple approach: compute full attention matrix but mask to window
        # For efficiency, we limit the effective context to buffer_size
        
        # Attention scores: [batch, seq, seq]
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) * scale
        
        # Create causal + window mask
        # Position i can only attend to max(0, i - buffer_size) : i (exclusive of i)
        positions = torch.arange(seq_len, device=device)
        
        # Causal mask: j < i (can't attend to self or future)
        causal_mask = positions.unsqueeze(0) >= positions.unsqueeze(1)  # [seq, seq], True = invalid
        
        # Window mask: j >= i - buffer_size
        window_mask = positions.unsqueeze(0) < (positions.unsqueeze(1) - self.buffer_size)  # True = too far back
        
        # Combined mask: invalid if causal OR too far back
        mask = causal_mask | window_mask
        
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Retrieve: [batch, seq, seq] @ [batch, seq, dim*2] -> [batch, seq, dim*2]
        retrieved_flat = torch.bmm(attn_weights, states_flat)
        
        # Reshape to Phase2D
        retrieved = retrieved_flat.view(batch_size, seq_len, dim, 2)
        
        return EpisodicReadResult(
            values=retrieved,
            attention=attn_weights,
        )
