#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Phase-Field LLM: Main model that wires all components

The model is configured via V4Config and uses the registry to
instantiate all components (banks, coupler, backbone, memory, etc.).

Supports two embedding modes:
- BPE: Standard subword tokenization with Phase2DEmbed
- Morphological: Root + prefix/suffix with rotation operators
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
from .core.morphology_embed import DualEmbedding
from .core.byte_patching import BytePatchingModule, BytePatchInfo
from .memory.episodic import EpisodicMemoryEfficient


@dataclass
class ModelOutput:
    """Output from forward pass"""
    logits: torch.Tensor  # [batch, seq, vocab_size] (root vocab for morphological)
    phase_states: torch.Tensor  # [batch, seq, dim, 2]
    bank_outputs: Dict[str, torch.Tensor]  # {bank_name: [batch, seq, dim, 2]}
    memory_result: Optional[MemoryReadResult] = None
    backbone_state: Optional[BackboneState] = None
    coupling_loss: Optional[torch.Tensor] = None
    # Morphological mode: prefix/suffix logits for full text generation
    prefix_logits: Optional[torch.Tensor] = None  # [batch, seq, prefix_vocab_size]
    suffix_logits: Optional[torch.Tensor] = None  # [batch, seq, suffix_vocab_size]
    # Byte patching info (for byte mode with patching)
    byte_patch_info: Optional[BytePatchInfo] = None
    # Metrics for manas/buddhi/viveka/smriti (to be populated)
    metrics: Optional[Dict[str, torch.Tensor]] = None


class QuantumPhaseFieldLLM(nn.Module):
    """
    Quantum Phase-Field Language Model.
    
    Architecture:
        Tokens -> Phase2D Embed -> Banks -> Backbone -> Memory -> Coupler -> LM Head
    
    All components are injectable via config.
    
    Embedding modes:
    - BPE: input_ids -> Phase2DEmbed -> Phase2D
    - Morphological: (root_ids, prefix_ids, suffix_ids) -> MorphologyAwareEmbed -> Phase2D
    """
    
    def __init__(self, config: V4Config):
        super().__init__()
        self.config = config
        registry = get_registry()
        
        # Embedding mode
        self.embedding_mode = config.tokenizer.mode
        
        # Check if using byte patching
        self.use_byte_patching = (
            config.tokenizer.mode == 'byte' and 
            config.tokenizer.byte_patching.enabled
        )
        
        # Byte patching module (used when tokenizer mode is 'byte' with patching)
        self.byte_patching: Optional[BytePatchingModule] = None
        if self.use_byte_patching:
            bp_config = config.tokenizer.byte_patching
            self.byte_patching = BytePatchingModule(
                vocab_size=config.vocab_size,  # Should be 259 for byte mode
                dim=config.dim,
                patch_size=bp_config.patch_size,
                decoder_layers=bp_config.decoder_layers,
                padding_idx=256,  # PAD token for byte mode
            )
        
        # Dual embedding: supports both BPE and morphological modes
        self.embed = DualEmbedding(
            bpe_vocab_size=config.vocab_size,
            root_vocab_size=config.tokenizer.root_vocab_size,
            prefix_vocab_size=config.tokenizer.prefix_vocab_size,
            suffix_vocab_size=config.tokenizer.suffix_vocab_size,
            dim=config.dim,
            mode=config.tokenizer.mode if config.tokenizer.mode != 'byte' else 'bpe',
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
            use_scan=getattr(config.backbone, 'use_scan', True),
            **config.backbone.params
        )
        
        # Memory (global learned slots)
        self.memory = registry.create_memory(
            config.memory.type,
            dim=config.dim,
            num_slots=config.memory.num_slots,
            **config.memory.params
        )
        
        # Episodic memory (per-sequence buffer for copy capability)
        self.use_episodic = config.episodic.enabled
        self.episodic_weight = config.episodic.weight
        if self.use_episodic:
            self.episodic = EpisodicMemoryEfficient(
                dim=config.dim,
                buffer_size=config.episodic.buffer_size,
            )
        else:
            self.episodic = None
        
        # LM Head: Phase2D -> logits
        self.lm_head = nn.Sequential(
            Phase2DLinear(config.dim, config.dim),
            Phase2DLayerNorm(config.dim),
        )
        self.output_proj = nn.Linear(config.dim * 2, config.vocab_size, bias=False)
        
        # Morphological heads: predict prefix and suffix for each position
        # These are only used in morphological mode for full text generation
        if config.tokenizer.mode == 'morphological':
            self.prefix_proj = nn.Linear(config.dim * 2, config.tokenizer.prefix_vocab_size, bias=False)
            self.suffix_proj = nn.Linear(config.dim * 2, config.tokenizer.suffix_vocab_size, bias=False)
        else:
            self.prefix_proj = None
            self.suffix_proj = None
        
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
        input_ids: Optional[torch.Tensor] = None,  # [batch, seq] for BPE mode
        root_ids: Optional[torch.Tensor] = None,   # [batch, seq] for morphological mode
        prefix_ids: Optional[torch.Tensor] = None, # [batch, seq] for morphological mode
        suffix_ids: Optional[torch.Tensor] = None, # [batch, seq] for morphological mode
        backbone_state: Optional[BackboneState] = None,
        use_memory: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        """
        Forward pass through the model.
        
        Supports two input modes:
        - BPE mode: provide input_ids
        - Morphological mode: provide root_ids, prefix_ids, suffix_ids
        
        Args:
            input_ids: Token IDs [batch, seq] (BPE mode)
            root_ids: Root token IDs [batch, seq] (morphological mode)
            prefix_ids: Prefix token IDs [batch, seq] (morphological mode)
            suffix_ids: Suffix token IDs [batch, seq] (morphological mode)
            backbone_state: Optional state for streaming
            use_memory: Whether to use memory module
            context: Optional context (language_id, etc.)
        
        Returns:
            ModelOutput with logits, phase states, etc.
        """
        context = context or {}
        byte_patch_info: Optional[BytePatchInfo] = None
        original_input_ids: Optional[torch.Tensor] = None
        
        # 1. Embed tokens to Phase2D (based on mode)
        if self.use_byte_patching and input_ids is not None:
            # Byte patching mode: convert bytes to patch latents
            original_input_ids = input_ids
            x, byte_patch_info = self.byte_patching.encode(input_ids)
            context['token_ids'] = input_ids
            context['byte_patch_info'] = byte_patch_info
        elif self.embedding_mode == 'morphological' and root_ids is not None:
            x = self.embed(
                root_ids=root_ids,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                mode='morphological',
            )
            context['token_ids'] = root_ids  # Use root IDs for downstream
            context['prefix_ids'] = prefix_ids
            context['suffix_ids'] = suffix_ids
        elif input_ids is not None:
            x = self.embed(input_ids=input_ids, mode='bpe')
            context['token_ids'] = input_ids
        else:
            raise ValueError("Either input_ids (BPE/byte) or root_ids/prefix_ids/suffix_ids (morphological) must be provided")
        
        # 2. Process through each bank
        bank_outputs = {}
        for bank_name, bank in self.banks.items():
            bank_out = bank(x, context=context)
            bank_outputs[bank_name] = bank_out
        
        # 3. Couple banks via interference (or bypass if bankless)
        # Bankless mode is useful as a baseline: embeddings -> backbone directly.
        if len(bank_outputs) == 0:
            coupled = x
        else:
            coupled = self.coupler(bank_outputs, context=context)
        
        # 4. Process through backbone
        backbone_out, new_backbone_state = self.backbone(coupled, state=backbone_state)
        
        # 5. Query global memory (if enabled)
        memory_result = None
        if use_memory:
            memory_result = self.memory.read(backbone_out, top_k=32)  # Reduced from 64 for speed
            # Add memory to backbone output
            backbone_out = backbone_out + memory_result.values * 0.1
        
        # 5b. Episodic memory: retrieve from recent positions in this sequence
        # This provides "copy" capability similar to transformer attention
        if self.episodic is not None:
            episodic_result = self.episodic(backbone_out)
            backbone_out = backbone_out + episodic_result.values * self.episodic_weight
        
        # 6. LM head (different paths for byte patching vs standard)
        prefix_logits = None
        suffix_logits = None
        
        if self.use_byte_patching and byte_patch_info is not None and original_input_ids is not None:
            # Byte patching mode: decode patch states to per-byte logits
            logits = self.byte_patching.decode(
                backbone_out, original_input_ids, byte_patch_info
            )  # [batch, T, vocab_size]
        else:
            # Standard LM head path
            lm_out = self.lm_head(backbone_out)
            
            # Convert Phase2D to real for final projection
            lm_real = phase2d_to_real(lm_out, mode='concat')  # [batch, seq, dim*2]
            logits = self.output_proj(lm_real)  # [batch, seq, vocab_size]
            
            # Morphological mode: also predict prefix and suffix
            if self.prefix_proj is not None and self.suffix_proj is not None:
                prefix_logits = self.prefix_proj(lm_real)  # [batch, seq, prefix_vocab_size]
                suffix_logits = self.suffix_proj(lm_real)  # [batch, seq, suffix_vocab_size]
        
        # Compute coupling loss (for training)
        coupling_loss = self.coupler.compute_coupling_loss(bank_outputs)
        
        # Compute philosophy metrics (manas/buddhi/viveka/smriti)
        metrics = None
        if context.get('compute_metrics', False):
            from .metrics.philosophy_metrics import compute_all_metrics
            philosophy_metrics = compute_all_metrics(
                logits=logits,
                phase_states=backbone_out,
                bank_outputs=bank_outputs,
                memory_result=memory_result,
                backbone_state=new_backbone_state,
                coupling_loss=coupling_loss,
                prev_bank_states=context.get('prev_bank_states'),
            )
            metrics = philosophy_metrics.to_dict()
        
        return ModelOutput(
            logits=logits,
            phase_states=backbone_out,
            bank_outputs=bank_outputs,
            memory_result=memory_result,
            backbone_state=new_backbone_state,
            coupling_loss=coupling_loss,
            prefix_logits=prefix_logits,
            suffix_logits=suffix_logits,
            byte_patch_info=byte_patch_info,
            metrics=metrics,
        )
    
    def set_embedding_mode(self, mode: str) -> None:
        """
        Switch embedding mode between 'bpe', 'morphological', and 'byte'.
        
        Args:
            mode: 'bpe', 'morphological', or 'byte'
        """
        if mode not in ['bpe', 'morphological', 'byte']:
            raise ValueError(f"Mode must be 'bpe', 'morphological', or 'byte', got {mode}")
        self.embedding_mode = mode
        # For byte mode, we use the byte patching path, not the dual embedding
        if mode != 'byte':
            self.embed.mode = mode
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        root_ids: Optional[torch.Tensor] = None,
        prefix_ids: Optional[torch.Tensor] = None,
        suffix_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
        bad_token_ids: Optional[List[int]] = None,
        sampler=None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Note: For morphological mode generation, you need a way to map
        generated root IDs back to (prefix, suffix) pairs. This is
        a simplified implementation that only generates root IDs.
        
        Args:
            input_ids: Prompt tokens [batch, seq] (BPE mode)
            root_ids: Root tokens [batch, seq] (morphological mode)
            prefix_ids: Prefix tokens [batch, seq] (morphological mode)
            suffix_ids: Suffix tokens [batch, seq] (morphological mode)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_token_id: Stop generation at this token
            bad_token_ids: Token IDs to filter out (set logits to -inf)
            sampler: Optional custom sampler
        
        Returns:
            Generated tokens [batch, seq + max_new_tokens]
            For BPE mode: generated input_ids
            For morphological mode: generated root_ids
        """
        self.eval()
        
        # Get default sampler
        if sampler is None:
            from .sampler import AutoregressiveSampler
            sampler = AutoregressiveSampler()
        
        # Determine mode and initialize
        if self.embedding_mode == 'morphological' and root_ids is not None:
            generated_roots = root_ids.clone()
            generated_prefixes = prefix_ids.clone() if prefix_ids is not None else None
            generated_suffixes = suffix_ids.clone() if suffix_ids is not None else None
            null_affix_id = 4  # Default null affix ID
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    output = self.forward(
                        root_ids=generated_roots,
                        prefix_ids=generated_prefixes,
                        suffix_ids=generated_suffixes,
                        use_memory=True,
                    )
                    
                    # Sample root token
                    next_root_logits = output.logits[:, -1, :].clone()
                    
                    # Filter out bad tokens (PAD, BOS, etc.)
                    if bad_token_ids:
                        for bad_id in bad_token_ids:
                            if bad_id < next_root_logits.size(-1):
                                next_root_logits[:, bad_id] = float('-inf')
                    
                    next_root, _ = sampler.sample(
                        next_root_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        past_tokens=generated_roots,
                    )
                    generated_roots = torch.cat([generated_roots, next_root], dim=1)
                    
                    # Sample prefix/suffix if heads are available
                    if output.prefix_logits is not None and generated_prefixes is not None:
                        next_prefix_logits = output.prefix_logits[:, -1, :]
                        next_prefix, _ = sampler.sample(
                            next_prefix_logits,
                            temperature=temperature,
                            top_k=min(top_k or 50, 20),  # Smaller top_k for affixes
                            top_p=top_p,
                        )
                        generated_prefixes = torch.cat([generated_prefixes, next_prefix], dim=1)
                    elif generated_prefixes is not None:
                        null_prefix = torch.full_like(next_root, null_affix_id)
                        generated_prefixes = torch.cat([generated_prefixes, null_prefix], dim=1)
                    
                    if output.suffix_logits is not None and generated_suffixes is not None:
                        next_suffix_logits = output.suffix_logits[:, -1, :]
                        next_suffix, _ = sampler.sample(
                            next_suffix_logits,
                            temperature=temperature,
                            top_k=min(top_k or 50, 20),  # Smaller top_k for affixes
                            top_p=top_p,
                        )
                        generated_suffixes = torch.cat([generated_suffixes, next_suffix], dim=1)
                    elif generated_suffixes is not None:
                        null_suffix = torch.full_like(next_root, null_affix_id)
                        generated_suffixes = torch.cat([generated_suffixes, null_suffix], dim=1)
                    
                    if eos_token_id is not None and (next_root == eos_token_id).all():
                        break
            
            # Return tuple of (roots, prefixes, suffixes) for full decoding
            return (generated_roots, generated_prefixes, generated_suffixes)
        
        elif self.use_byte_patching and input_ids is not None:
            # Byte patching mode: generate patch-by-patch
            return self._generate_byte_patching(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                bad_token_ids=bad_token_ids,
                sampler=sampler,
            )
        
        else:
            # BPE mode (also handles byte mode without patching)
            if input_ids is None:
                raise ValueError("input_ids required for BPE/byte mode generation")
            
            generated = input_ids.clone()
            backbone_state = None
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    output = self.forward(
                        input_ids=generated,
                        backbone_state=backbone_state,
                        use_memory=True,
                    )
                    
                    next_logits = output.logits[:, -1, :]
                    next_token, _ = sampler.sample(
                        next_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        past_tokens=generated,
                    )
                    
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    if eos_token_id is not None and (next_token == eos_token_id).all():
                        break
            
            return generated
    
    def _generate_byte_patching(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        eos_token_id: Optional[int] = None,
        bad_token_ids: Optional[List[int]] = None,
        sampler=None,
    ) -> torch.Tensor:
        """
        Generate bytes using patch-by-patch decoding.
        
        Strategy:
        1. Encode prompt bytes to patch latents
        2. Process through backbone/memory to get patch states
        3. For each new patch, generate P bytes autoregressively
        4. Append new bytes and continue
        
        Args:
            input_ids: [batch, T] prompt byte IDs
            max_new_tokens: Maximum bytes to generate
            ... (other sampling parameters)
        
        Returns:
            [batch, T + generated_len] generated byte IDs
        """
        if sampler is None:
            from .sampler import AutoregressiveSampler
            sampler = AutoregressiveSampler()
        
        device = input_ids.device
        batch_size = input_ids.shape[0]
        patch_size = self.byte_patching.patch_size
        
        # Calculate number of new patches to generate
        num_new_patches = (max_new_tokens + patch_size - 1) // patch_size
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for patch_idx in range(num_new_patches):
                # Encode current bytes to patch latents
                patch_latents, info = self.byte_patching.encode(generated)
                
                # Process through banks/backbone/memory
                bank_outputs = {}
                for bank_name, bank in self.banks.items():
                    bank_out = bank(patch_latents, context={})
                    bank_outputs[bank_name] = bank_out
                
                if len(bank_outputs) == 0:
                    coupled = patch_latents
                else:
                    coupled = self.coupler(bank_outputs, context={})
                backbone_out, _ = self.backbone(coupled, state=None)
                
                # Query memory
                memory_result = self.memory.read(backbone_out, top_k=64)
                backbone_out = backbone_out + memory_result.values * 0.1
                
                # Get the last patch state for generation
                last_patch_state = backbone_out[:, -1, :, :]  # [batch, dim, 2]
                
                # Get the last byte from current sequence
                prev_byte = generated[:, -1]  # [batch]
                
                # Generate P new bytes for this patch
                new_bytes = self.byte_patching.decoder.generate_patch(
                    last_patch_state, prev_byte
                )  # [batch, P]
                
                # Apply sampling with temperature/top-k/top-p to each byte
                # (The generate_patch method uses greedy; we'll use sampler for better quality)
                sampled_bytes = []
                current_byte = prev_byte
                
                for p in range(patch_size):
                    # Get logits for this position
                    byte_embed = self.byte_patching.decoder.byte_embed(current_byte.unsqueeze(1))
                    byte_embed = byte_embed.squeeze(1)  # [batch, dim, 2]
                    
                    pos_embed = self.byte_patching.decoder.position_embed[p]
                    byte_embed = byte_embed + pos_embed
                    
                    x = byte_embed + last_patch_state
                    
                    for layer in self.byte_patching.decoder.layers:
                        residual = x
                        x = layer['proj'](x)
                        x = layer['norm'](x)
                        x = x + residual
                    
                    x = self.byte_patching.decoder.output_proj(x)
                    x = self.byte_patching.decoder.output_norm(x)
                    
                    from .core.phase2d import phase2d_to_real
                    x_real = phase2d_to_real(x, mode='concat')
                    logits = self.byte_patching.decoder.lm_head(x_real)
                    
                    # Filter bad tokens
                    if bad_token_ids:
                        for bad_id in bad_token_ids:
                            if bad_id < logits.size(-1):
                                logits[:, bad_id] = float('-inf')
                    
                    # Sample
                    next_byte, _ = sampler.sample(
                        logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    next_byte = next_byte.squeeze(-1)  # [batch]
                    sampled_bytes.append(next_byte)
                    current_byte = next_byte
                    
                    # Check for EOS
                    if eos_token_id is not None and (next_byte == eos_token_id).all():
                        break
                
                # Append sampled bytes
                new_bytes_tensor = torch.stack(sampled_bytes, dim=1)  # [batch, <=P]
                generated = torch.cat([generated, new_bytes_tensor], dim=1)
                
                # Check for EOS in any position
                if eos_token_id is not None:
                    if (generated[:, -1] == eos_token_id).all():
                        break
                
                # Stop if we've generated enough
                if generated.shape[1] >= input_ids.shape[1] + max_new_tokens:
                    break
        
        # Trim to max_new_tokens
        max_len = input_ids.shape[1] + max_new_tokens
        if generated.shape[1] > max_len:
            generated = generated[:, :max_len]
        
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
