#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Core Interfaces: Base classes for all swappable modules

All components implement these interfaces so they can be:
1. Registered in the registry
2. Instantiated from config
3. Swapped without changing the model wiring

Each interface defines:
- Required methods
- Expected input/output shapes
- Metadata for registry
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


# =============================================================================
# PhaseBank: One layer of the phase space (semantic/context/language/emotion/...)
# =============================================================================

class PhaseBank(ABC):
    """
    Base class for a phase-space layer (bank).
    
    Each bank maintains its own state and update rule.
    Banks are combined by the Coupler via interference.
    
    Input: Phase2D embeddings [batch, seq, dim, 2]
    Output: Phase2D bank state [batch, seq, dim, 2]
    
    Note: Implementations should also inherit from nn.Module.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this bank type"""
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Phase dimension of this bank"""
        pass
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor,  # [batch, seq, dim, 2] Phase2D
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Process input through this bank.
        
        Args:
            x: Phase2D input embeddings
            context: Optional context dict (token_ids, language_id, etc.)
        
        Returns:
            Phase2D bank output
        """
        pass
    
    def get_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get internal state for inspection/persistence"""
        return None
    
    def set_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Set internal state (for loading/continuing)"""
        pass


# =============================================================================
# Coupler: Combines multiple banks via interference
# =============================================================================

class Coupler(ABC):
    """
    Base class for coupling multiple phase banks.
    
    Takes outputs from all banks and combines them into a single
    Phase2D representation via interference-like operations.
    
    Must be GPU-friendly: GEMM-based, no trig in hot path.
    
    Note: Implementations should also inherit from nn.Module.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this coupler type"""
        pass
    
    @abstractmethod
    def forward(
        self,
        bank_outputs: Dict[str, torch.Tensor],  # {bank_name: [batch, seq, dim, 2]}
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Combine bank outputs via interference.
        
        Args:
            bank_outputs: Dict mapping bank names to their Phase2D outputs
            context: Optional context for conditional coupling
        
        Returns:
            [batch, seq, dim, 2] combined Phase2D output
        """
        pass
    
    def compute_coupling_loss(
        self,
        bank_outputs: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Optional: compute cross-bank agreement/coherence loss.
        Returns None if this coupler doesn't provide such a loss.
        """
        return None


# =============================================================================
# Backbone: Linear-time sequence processing
# =============================================================================

@dataclass
class BackboneState:
    """State for streaming/incremental backbone processing"""
    hidden: torch.Tensor  # Hidden state tensor
    step: int = 0  # Current step in sequence
    extra: Optional[Dict[str, torch.Tensor]] = None


class Backbone(ABC):
    """
    Base class for sequence backbone (SSM, hybrid, etc.)
    
    Processes Phase2D sequences with linear-time complexity.
    Supports streaming for long context (256K+).
    
    Note: Implementations should also inherit from nn.Module.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this backbone type"""
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Phase dimension"""
        pass
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Hidden state dimension"""
        pass
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, dim, 2] Phase2D
        state: Optional[BackboneState] = None
    ) -> Tuple[torch.Tensor, BackboneState]:
        """
        Process sequence through backbone.
        
        Args:
            x: Phase2D input
            state: Optional previous state for streaming
        
        Returns:
            (output, new_state): Phase2D output and updated state
        """
        pass
    
    def init_state(self, batch_size: int, device: torch.device) -> BackboneState:
        """Initialize state for streaming"""
        return BackboneState(
            hidden=torch.zeros(batch_size, self.state_dim, 2, device=device),
            step=0
        )


# =============================================================================
# Memory: Long-term phase-coded associative memory
# =============================================================================

@dataclass
class MemoryReadResult:
    """Result of memory read operation"""
    values: torch.Tensor  # Retrieved values [batch, seq, dim, 2]
    attention: torch.Tensor  # Attention weights [batch, seq, num_slots]
    hit_mask: Optional[torch.Tensor] = None  # Which slots were accessed


class Memory(ABC):
    """
    Base class for long-term memory.
    
    Stores phase-coded key-value pairs.
    Supports incremental learning (add/edit without retraining backbone).
    
    Note: Implementations should also inherit from nn.Module.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this memory type"""
        pass
    
    @property
    @abstractmethod
    def num_slots(self) -> int:
        """Number of memory slots"""
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Phase dimension"""
        pass
    
    @abstractmethod
    def read(
        self,
        query: torch.Tensor,  # [batch, seq, dim, 2] Phase2D query
        top_k: Optional[int] = None
    ) -> MemoryReadResult:
        """
        Read from memory using phase-coded query.
        
        Args:
            query: Phase2D query vectors
            top_k: Optional limit on number of slots to attend to
        
        Returns:
            MemoryReadResult with retrieved values and attention
        """
        pass
    
    @abstractmethod
    def write(
        self,
        key: torch.Tensor,    # [batch, dim, 2] Phase2D key
        value: torch.Tensor,  # [batch, dim, 2] Phase2D value
        importance: Optional[torch.Tensor] = None
    ) -> None:
        """
        Write to memory (during training or incremental learning).
        
        Args:
            key: Phase2D key to store
            value: Phase2D value to associate with key
            importance: Optional importance score for consolidation
        """
        pass
    
    def consolidate(self) -> None:
        """Consolidate memory (merge, prune, etc.) - called periodically"""
        pass
    
    def get_shard(self, shard_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get a memory shard for saving/transfer"""
        return None
    
    def add_shard(self, shard_id: str, shard: Dict[str, torch.Tensor]) -> None:
        """Add a memory shard (for incremental learning)"""
        pass


# =============================================================================
# Objective: Loss computation
# =============================================================================

@dataclass
class ObjectiveResult:
    """Result of objective computation"""
    loss: torch.Tensor  # Scalar loss
    metrics: Dict[str, float]  # Logged metrics
    

class Objective(ABC):
    """
    Base class for loss objectives.
    
    Each objective computes one loss term.
    Multiple objectives are combined by the training loop.
    
    Note: Implementations should also inherit from nn.Module.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this objective"""
        pass
    
    @property
    def weight(self) -> float:
        """Weight for combining with other objectives (can be overridden)"""
        return 1.0
    
    @abstractmethod
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        context: Optional[Dict[str, Any]] = None
    ) -> ObjectiveResult:
        """
        Compute loss.
        
        Args:
            model_output: Dict containing model outputs (logits, phase states, etc.)
            targets: Dict containing targets (token_ids, etc.)
            context: Optional context
        
        Returns:
            ObjectiveResult with loss and metrics
        """
        pass


# =============================================================================
# Sampler: Generation strategy
# =============================================================================

@dataclass
class SampleResult:
    """Result of sampling"""
    tokens: torch.Tensor  # Generated token ids [batch, gen_len]
    logprobs: Optional[torch.Tensor] = None  # Log probabilities
    phase_states: Optional[torch.Tensor] = None  # Phase states during generation


class Sampler(ABC):
    """
    Base class for generation samplers.
    
    Defines how tokens are sampled from model outputs.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this sampler"""
        pass
    
    @abstractmethod
    def sample(
        self,
        logits: torch.Tensor,  # [batch, vocab_size]
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample next token from logits.
        
        Args:
            logits: Logits from model
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
        
        Returns:
            (token_ids, log_probs): Sampled tokens and their log probabilities
        """
        pass
