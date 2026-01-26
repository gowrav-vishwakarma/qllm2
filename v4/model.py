#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Phase-Field LLM: Main model that wires all components

The model is configured via V4Config and uses the registry to
instantiate all components (banks, coupler, backbone, memory, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .core.config import V4Config, get_default_config
from .core.registry import get_registry
from .core.interfaces import BackboneState, MemoryReadResult
from .core.phase2d import Phase2DEmbed, Phase2DLinear, Phase2DLayerNorm, phase2d_to_real


@dataclass
class ModelOutput:
    """Output from forward pass"""
    logits: torch.Tensor  # [batch, seq, vocab_size]
    phase_states: torch.Tensor  # [batch, seq, dim, 2]
    bank_outputs: Dict[str, torch.Tensor]  # {bank_name: [batch, seq, dim, 2]}
    memory_result: Optional[MemoryReadResult] = None
    backbone_state: Optional[BackboneState] = None
    coupling_loss: Optional[torch.Tensor] = None


class QuantumPhaseFieldLLM(nn.Module):
    """
    Quantum Phase-Field Language Model.
    
    Architecture:
        Tokens -> Phase2D Embed -> Banks -> Backbone -> Memory -> Coupler -> LM Head
    
    All components are injectable via config.
    """
    
    def __init__(self, config: V4Config):
        super().__init__()
        self.config = config
        registry = get_registry()
        
        # Embedding: tokens -> Phase2D
        self.embed = Phase2DEmbed(
            vocab_size=config.vocab_size,
            dim=config.dim,
            padding_idx=0,  # Assume 0 is padding
        )
        
        # Phase banks (separate layers)
        self.banks = nn.ModuleDict()
        for bank_name, bank_cfg in config.banks.items():
            self.banks[bank_name] = registry.create_bank(
                bank_cfg.type,
                dim=bank_cfg.dim,
                **bank_cfg.params
            )
        
        # Coupler
        self.coupler = registry.create_coupler(
            config.coupler.type,
            dim=config.dim,
            bank_names=list(config.banks.keys()),
            **config.coupler.params
        )
        
        # Backbone
        self.backbone = registry.create_backbone(
            config.backbone.type,
            dim=config.dim,
            state_dim=config.backbone.state_dim,
            num_layers=config.backbone.num_layers,
            **config.backbone.params
        )
        
        # Memory
        self.memory = registry.create_memory(
            config.memory.type,
            dim=config.dim,
            num_slots=config.memory.num_slots,
            **config.memory.params
        )
        
        # LM Head: Phase2D -> logits
        self.lm_head = nn.Sequential(
            Phase2DLinear(config.dim, config.dim),
            Phase2DLayerNorm(config.dim),
        )
        self.output_proj = nn.Linear(config.dim * 2, config.vocab_size, bias=False)
        
        # Tie embeddings with output projection
        # (Optional: enable with config flag)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq]
        backbone_state: Optional[BackboneState] = None,
        use_memory: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch, seq]
            backbone_state: Optional state for streaming
            use_memory: Whether to use memory module
            context: Optional context (language_id, etc.)
        
        Returns:
            ModelOutput with logits, phase states, etc.
        """
        context = context or {}
        context['token_ids'] = input_ids
        
        # 1. Embed tokens to Phase2D
        x = self.embed(input_ids)  # [batch, seq, dim, 2]
        
        # 2. Process through each bank
        bank_outputs = {}
        for bank_name, bank in self.banks.items():
            bank_out = bank(x, context=context)
            bank_outputs[bank_name] = bank_out
        
        # 3. Couple banks via interference
        coupled = self.coupler(bank_outputs, context=context)
        
        # 4. Process through backbone
        backbone_out, new_backbone_state = self.backbone(coupled, state=backbone_state)
        
        # 5. Query memory (if enabled)
        memory_result = None
        if use_memory:
            memory_result = self.memory.read(backbone_out, top_k=64)
            # Add memory to backbone output
            backbone_out = backbone_out + memory_result.values * 0.1
        
        # 6. LM head
        lm_out = self.lm_head(backbone_out)
        
        # Convert Phase2D to real for final projection
        lm_real = phase2d_to_real(lm_out, mode='concat')  # [batch, seq, dim*2]
        logits = self.output_proj(lm_real)  # [batch, seq, vocab_size]
        
        # Compute coupling loss (for training)
        coupling_loss = self.coupler.compute_coupling_loss(bank_outputs)
        
        return ModelOutput(
            logits=logits,
            phase_states=backbone_out,
            bank_outputs=bank_outputs,
            memory_result=memory_result,
            backbone_state=new_backbone_state,
            coupling_loss=coupling_loss,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
        sampler=None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Prompt tokens [batch, seq]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_token_id: Stop generation at this token
            sampler: Optional custom sampler
        
        Returns:
            Generated tokens [batch, seq + max_new_tokens]
        """
        self.eval()
        
        # Get default sampler
        if sampler is None:
            from .sampler import AutoregressiveSampler
            sampler = AutoregressiveSampler()
        
        generated = input_ids.clone()
        backbone_state = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                output = self.forward(
                    generated,
                    backbone_state=backbone_state,
                    use_memory=True,
                )
                
                next_logits = output.logits[:, -1, :]  # [batch, vocab_size]
                
                # Sample next token
                next_token, _ = sampler.sample(
                    next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    past_tokens=generated,
                )
                
                # Append to generated
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if eos_token_id is not None:
                    if (next_token == eos_token_id).all():
                        break
        
        return generated
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component"""
        counts = {
            'embed': sum(p.numel() for p in self.embed.parameters()),
            'banks': sum(p.numel() for p in self.banks.parameters()),
            'coupler': sum(p.numel() for p in self.coupler.parameters()),
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'memory': sum(p.numel() for p in self.memory.parameters()),
            'lm_head': sum(p.numel() for p in self.lm_head.parameters()) + 
                       sum(p.numel() for p in self.output_proj.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


def create_model(
    config: Optional[V4Config] = None,
    size: str = 'small',
    **overrides
) -> QuantumPhaseFieldLLM:
    """
    Create a model from config or preset size.
    
    Args:
        config: Full config (takes precedence)
        size: Preset size ('tiny', 'small', 'medium', 'large')
        **overrides: Override specific config values
    
    Returns:
        Initialized model
    """
    if config is None:
        config = get_default_config(size)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return QuantumPhaseFieldLLM(config)
