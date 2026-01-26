#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoregressive Sampler: Standard token-by-token generation
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from ..core.interfaces import Sampler
from ..core.registry import register_sampler


@register_sampler('autoregressive', description='Standard autoregressive sampling with temperature/top-k/top-p')
class AutoregressiveSampler(Sampler):
    """
    Standard autoregressive sampler with various decoding strategies.
    """
    
    def __init__(
        self,
        default_temperature: float = 1.0,
        default_top_k: Optional[int] = 50,
        default_top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0,
        min_p: Optional[float] = None,
    ):
        self.default_temperature = default_temperature
        self.default_top_k = default_top_k
        self.default_top_p = default_top_p
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p
    
    @property
    def name(self) -> str:
        return "autoregressive"
    
    def sample(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        past_tokens: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample next token from logits.
        
        Args:
            logits: [batch, vocab_size] logits
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            past_tokens: Previous tokens for repetition penalty
        
        Returns:
            (token_ids, log_probs): [batch, 1] sampled tokens and their log probs
        """
        temperature = temperature or self.default_temperature
        top_k = top_k if top_k is not None else self.default_top_k
        top_p = top_p if top_p is not None else self.default_top_p
        
        # Apply repetition penalty
        if past_tokens is not None and self.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, past_tokens)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            logits = self._top_k_filter(logits, top_k)
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            logits = self._top_p_filter(logits, top_p)
        
        # Apply min-p filtering
        if self.min_p is not None:
            logits = self._min_p_filter(logits, self.min_p)
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1)
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, token_ids)
        
        return token_ids, selected_log_probs
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        past_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Apply repetition penalty to discourage repeated tokens."""
        for i in range(logits.size(0)):
            for token_id in past_tokens[i].unique():
                if token_id >= 0:  # Skip padding/special tokens
                    if logits[i, token_id] > 0:
                        logits[i, token_id] /= self.repetition_penalty
                    else:
                        logits[i, token_id] *= self.repetition_penalty
        return logits
    
    def _top_k_filter(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Keep only top-k tokens."""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus sampling: keep tokens with cumulative probability < top_p."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _min_p_filter(self, logits: torch.Tensor, min_p: float) -> torch.Tensor:
        """Min-p sampling: keep tokens with probability >= min_p * max_prob."""
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1, keepdim=True)[0]
        threshold = min_p * max_probs
        logits[probs < threshold] = float('-inf')
        return logits
