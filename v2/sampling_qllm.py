#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampling methods for Quantum-Inspired LLM
"""

import torch
import torch.nn.functional as F
from typing import List, Optional

def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits"""
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits"""
    if p >= 1.0:
        return logits
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float('-inf'))

def apply_repetition_penalty(logits: torch.Tensor, recent_ids: List[int], penalty: float) -> torch.Tensor:
    """Apply repetition penalty to logits"""
    if penalty == 1.0 or len(recent_ids) == 0:
        return logits
    
    for token_id in set(recent_ids):
        logits[..., token_id] /= penalty
    
    return logits

def sample_next_token(logits: torch.Tensor,
                      temperature: float = 1.0,
                      top_k: int = 0,
                      top_p: float = 1.0,
                      repetition_penalty: float = 1.0,
                      recent_ids: Optional[List[List[int]]] = None,
                      min_p: float = 0.0) -> torch.Tensor:
    """Sample next token from logits"""
    
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply repetition penalty
    if recent_ids is not None and repetition_penalty != 1.0:
        recent_tokens = recent_ids[0] if isinstance(recent_ids, list) else []
        logits = apply_repetition_penalty(logits, recent_tokens, repetition_penalty)
    
    # Apply top-k filtering
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    
    # Apply top-p filtering
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    
    # Apply min-p filtering
    if min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        min_prob = probs.max(dim=-1, keepdim=True).values * min_p
        logits = torch.where(probs < min_prob, torch.full_like(logits, float('-inf')), logits)
    
    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token.squeeze(1)