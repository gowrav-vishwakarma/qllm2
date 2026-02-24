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
    Episodic memory using PyTorch SDPA for hardware-accelerated attention.
    
    Uses torch.nn.functional.scaled_dot_product_attention which automatically
    dispatches to FlashAttention or memory-efficient backends. Sliding window
    is implemented by processing in chunks of buffer_size.
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
        self.head_dim = dim * 2
        
        # Lightweight projections
        self.query_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        self.key_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        self.value_proj = nn.Linear(dim * 2, dim * 2, bias=False)
        
        nn.init.eye_(self.query_proj.weight)
        nn.init.eye_(self.key_proj.weight)
        nn.init.eye_(self.value_proj.weight)
    
    def forward(
        self,
        states: torch.Tensor,  # [batch, seq, dim, 2]
    ) -> EpisodicReadResult:
        """
        Efficient episodic retrieval using SDPA with sliding window via chunking.
        
        For each chunk of buffer_size tokens, we gather the relevant key-value
        context (current chunk + preceding buffer_size tokens) and run causal
        SDPA. This keeps memory at O(n * buffer_size) instead of O(n^2).
        """
        batch_size, seq_len, dim, _ = states.shape
        
        if seq_len <= 1:
            return EpisodicReadResult(
                values=torch.zeros_like(states),
                attention=None,
            )
        
        # Flatten Phase2D: [batch, seq, dim*2]
        states_flat = states.view(batch_size, seq_len, dim * 2)
        
        queries = self.query_proj(states_flat)
        keys = self.key_proj(states_flat)
        values = self.value_proj(states_flat)
        
        # Reshape for SDPA: [batch, 1 head, seq, head_dim]
        q = queries.unsqueeze(1)
        k = keys.unsqueeze(1)
        v = values.unsqueeze(1)
        
        if seq_len <= self.buffer_size * 2:
            # Short sequences: use causal SDPA directly (efficient enough)
            retrieved_flat = F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            ).squeeze(1)  # [batch, seq, dim*2]
        else:
            # Longer sequences: chunk-based sliding window
            chunk_size = self.buffer_size
            retrieved_chunks = []
            
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                # Context window: from max(0, start - buffer_size) to end
                ctx_start = max(0, start - self.buffer_size)
                
                q_chunk = q[:, :, start:end, :]  # [batch, 1, chunk, dim*2]
                k_ctx = k[:, :, ctx_start:end, :]
                v_ctx = v[:, :, ctx_start:end, :]
                
                # Build causal mask for this chunk relative to context
                q_len = end - start
                kv_len = end - ctx_start
                # Each query at position i (0-indexed within chunk) can attend
                # to context positions j where j < (start - ctx_start + i)
                attn_mask = torch.zeros(q_len, kv_len, device=states.device, dtype=states.dtype)
                for qi in range(q_len):
                    # Number of valid KV positions for this query
                    valid_kv = (start - ctx_start) + qi  # positions before this query
                    if valid_kv < kv_len:
                        attn_mask[qi, valid_kv:] = float('-inf')
                
                chunk_out = F.scaled_dot_product_attention(
                    q_chunk, k_ctx, v_ctx, attn_mask=attn_mask.unsqueeze(0).unsqueeze(0)
                ).squeeze(1)  # [batch, chunk, dim*2]
                
                retrieved_chunks.append(chunk_out)
            
            retrieved_flat = torch.cat(retrieved_chunks, dim=1)  # [batch, seq, dim*2]
        
        # First position has nothing to attend to, zero it out
        retrieved_flat[:, 0, :] = 0.0
        
        retrieved = retrieved_flat.view(batch_size, seq_len, dim, 2)
        
        return EpisodicReadResult(
            values=retrieved,
            attention=None,
        )
