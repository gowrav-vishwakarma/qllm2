#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Entropy Objective: Standard language modeling loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from ..core.interfaces import Objective, ObjectiveResult
from ..core.registry import register_objective


@register_objective('ce', description='Cross-entropy loss for next-token prediction')
class CrossEntropyObjective(nn.Module):
    """
    Standard cross-entropy loss for language modeling.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self._weight = weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    @property
    def name(self) -> str:
        return "ce"
    
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
        Compute cross-entropy loss.
        
        Args:
            model_output: Must contain 'logits' [batch, seq, vocab_size]
            targets: Must contain 'token_ids' [batch, seq]
        """
        logits = model_output['logits']  # [batch, seq, vocab_size]
        target_ids = targets['token_ids']  # [batch, seq]
        
        # Shift for next-token prediction
        # logits: predict token at position i+1
        # targets: token at position i+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()
        
        # Flatten for loss computation
        loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1)
        )
        
        # Compute perplexity for logging
        with torch.no_grad():
            perplexity = torch.exp(loss).item()
        
        return ObjectiveResult(
            loss=loss,
            metrics={
                'ce_loss': loss.item(),
                'perplexity': perplexity,
            }
        )
