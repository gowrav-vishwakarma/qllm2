#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Config: Configuration system for the model

Supports:
- YAML/JSON config files
- Programmatic config construction
- Validation and defaults
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


@dataclass
class BankConfig:
    """Configuration for a single PhaseBank"""
    type: str  # Registry name
    dim: int = 256
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CouplerConfig:
    """Configuration for the Coupler"""
    type: str = "interference"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackboneConfig:
    """Configuration for the Backbone"""
    type: str = "oscillatory_ssm"
    dim: int = 256
    state_dim: int = 512
    num_layers: int = 8
    use_scan: bool = True  # Enable vectorized scan for better speed
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Configuration for Memory"""
    type: str = "phase_associative"
    num_slots: int = 4096
    dim: int = 256
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodicConfig:
    """Configuration for Episodic Memory (per-sequence copy buffer)"""
    enabled: bool = True  # Enable episodic memory for copy capability
    buffer_size: int = 64  # Number of positions to remember
    weight: float = 0.1  # How much episodic retrieval influences output


@dataclass
class ObjectiveConfig:
    """Configuration for a single Objective"""
    type: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplerConfig:
    """Configuration for Sampler"""
    type: str = "autoregressive"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    max_steps: Optional[int] = None
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints_v4"
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    # Speed optimizations
    compile_model: bool = False  # Enable torch.compile
    compile_mode: str = "reduce-overhead"  # compile mode: "default", "reduce-overhead", "max-autotune"
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    prefetch_factor: int = 2  # DataLoader prefetch factor
    use_token_cache: bool = True  # Cache tokenized samples
    compute_metrics: bool = True  # Compute philosophy metrics during training


@dataclass
class BytePatchingConfig:
    """Configuration for byte patching (used when tokenizer mode is 'byte')"""
    enabled: bool = True  # Enable byte patching when using byte tokenizer
    patch_size: int = 4  # Number of bytes per patch
    decoder_layers: int = 2  # Number of layers in within-patch byte decoder
    decoder_dim: Optional[int] = None  # Decoder hidden dim (defaults to config.dim)


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer"""
    mode: str = 'bpe'  # 'bpe', 'morphological', or 'byte'
    bpe_name: str = 'gpt2'  # Model name for BPE tokenizer
    # Morphological tokenizer settings
    root_vocab_size: int = 16000
    prefix_vocab_size: int = 2000
    suffix_vocab_size: int = 2000
    morph_path: Optional[str] = None  # Path to save/load morphological tokenizer
    # Byte patching settings
    byte_patching: BytePatchingConfig = field(default_factory=BytePatchingConfig)


@dataclass
class V4Config:
    """
    Full v4 model configuration.
    
    Defines the complete model architecture via the injectable components.
    """
    # Model dimensions
    vocab_size: int = 50257  # BPE vocab size (used when mode='bpe')
    dim: int = 256  # Base phase dimension
    max_seq_len: int = 262144  # 256K context
    
    # Tokenizer configuration
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    
    # Phase banks (separate layers)
    banks: Dict[str, BankConfig] = field(default_factory=lambda: {
        'semantic': BankConfig(type='semantic', dim=256),
        'context': BankConfig(type='context', dim=256),
    })
    
    # Coupler
    coupler: CouplerConfig = field(default_factory=CouplerConfig)
    
    # Backbone
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    
    # Memory (global learned slots)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    
    # Episodic memory (per-sequence buffer for copy capability)
    episodic: EpisodicConfig = field(default_factory=EpisodicConfig)
    
    # Objectives
    objectives: List[ObjectiveConfig] = field(default_factory=lambda: [
        ObjectiveConfig(type='ce', weight=1.0),
        ObjectiveConfig(type='coherence', weight=0.01),
        ObjectiveConfig(type='coupling', weight=0.1),
    ])
    
    # Sampler
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    
    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save config to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'V4Config':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'V4Config':
        """Create config from dictionary"""
        # Convert nested dicts to dataclass instances
        if 'banks' in data:
            data['banks'] = {
                name: BankConfig(**cfg) if isinstance(cfg, dict) else cfg
                for name, cfg in data['banks'].items()
            }
        
        if 'coupler' in data and isinstance(data['coupler'], dict):
            data['coupler'] = CouplerConfig(**data['coupler'])
        
        if 'backbone' in data and isinstance(data['backbone'], dict):
            data['backbone'] = BackboneConfig(**data['backbone'])
        
        if 'memory' in data and isinstance(data['memory'], dict):
            data['memory'] = MemoryConfig(**data['memory'])
        
        if 'episodic' in data and isinstance(data['episodic'], dict):
            data['episodic'] = EpisodicConfig(**data['episodic'])
        
        if 'objectives' in data:
            data['objectives'] = [
                ObjectiveConfig(**obj) if isinstance(obj, dict) else obj
                for obj in data['objectives']
            ]
        
        if 'sampler' in data and isinstance(data['sampler'], dict):
            data['sampler'] = SamplerConfig(**data['sampler'])
        
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        
        if 'tokenizer' in data and isinstance(data['tokenizer'], dict):
            tok_data = data['tokenizer']
            # Handle nested byte_patching config
            if 'byte_patching' in tok_data and isinstance(tok_data['byte_patching'], dict):
                tok_data['byte_patching'] = BytePatchingConfig(**tok_data['byte_patching'])
            data['tokenizer'] = TokenizerConfig(**tok_data)
        
        return cls(**data)


def load_config(path: Union[str, Path]) -> V4Config:
    """Load config from file (JSON or YAML)"""
    path = Path(path)
    
    if path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
    else:
        with open(path, 'r') as f:
            data = json.load(f)
    
    return V4Config.from_dict(data)


def get_default_config(size: str = 'small') -> V4Config:
    """
    Get a default configuration for a given model size.
    
    Sizes (standard - 4 banks for rich linguistic modeling):
    - 'tiny': For testing (~1M params)
    - 'small': For quick experiments (~10M params)
    - 'medium': Balanced (~50M params)
    - 'large': Production (~200M params)
    
    Sizes (byte-optimized - 2 banks for speed, use with byte tokenizer):
    - 'tiny-byte': Minimal byte model
    - 'small-byte': Fast byte experiments
    - 'medium-byte': Balanced byte model (recommended for RTX 4090)
    - 'large-byte': Production byte model
    """
    configs = {
        'tiny': V4Config(
            dim=64,
            backbone=BackboneConfig(dim=64, state_dim=128, num_layers=4),
            memory=MemoryConfig(dim=64, num_slots=512),
            banks={
                'semantic': BankConfig(type='semantic', dim=64),
            },
            training=TrainingConfig(batch_size=16, learning_rate=1e-3),
        ),
        'small': V4Config(
            dim=256,
            backbone=BackboneConfig(dim=256, state_dim=512, num_layers=8),
            memory=MemoryConfig(dim=256, num_slots=512),  # Reduced for consumer GPUs
            banks={
                'semantic': BankConfig(type='semantic', dim=256),
                'context': BankConfig(type='context', dim=256),
                'morphology': BankConfig(type='morphology', dim=256),
                'orthography': BankConfig(type='orthography', dim=256),
            },
            training=TrainingConfig(batch_size=8, learning_rate=1e-4),
        ),
        'medium': V4Config(
            dim=512,
            backbone=BackboneConfig(dim=512, state_dim=1024, num_layers=12),
            memory=MemoryConfig(dim=512, num_slots=1024),  # Reduced for consumer GPUs
            banks={
                'semantic': BankConfig(type='semantic', dim=512),
                'context': BankConfig(type='context', dim=512),
                'morphology': BankConfig(type='morphology', dim=512),
                'orthography': BankConfig(type='orthography', dim=512),
            },
            training=TrainingConfig(batch_size=4, learning_rate=5e-5),
        ),
        'large': V4Config(
            dim=768,
            backbone=BackboneConfig(dim=768, state_dim=1536, num_layers=16),
            memory=MemoryConfig(dim=768, num_slots=8192),
            banks={
                'semantic': BankConfig(type='semantic', dim=768),
                'context': BankConfig(type='context', dim=768),
                'language': BankConfig(type='language', dim=768),
                'emotion': BankConfig(type='emotion', dim=768),
            },
            training=TrainingConfig(batch_size=2, learning_rate=3e-5),
        ),
        # =====================================================================
        # Byte-optimized configs: 2 banks only (semantic + context)
        # Faster, lower VRAM, better for byte tokenizer where morphology/ortho
        # banks provide diminishing returns on raw byte sequences.
        # =====================================================================
        'tiny-byte': V4Config(
            dim=64,
            backbone=BackboneConfig(dim=64, state_dim=128, num_layers=4),
            memory=MemoryConfig(dim=64, num_slots=256),
            banks={
                'semantic': BankConfig(type='semantic', dim=64),
                'context': BankConfig(type='context', dim=64),
            },
            training=TrainingConfig(batch_size=32, learning_rate=1e-3),
        ),
        'small-byte': V4Config(
            dim=256,
            backbone=BackboneConfig(dim=256, state_dim=512, num_layers=8),
            memory=MemoryConfig(dim=256, num_slots=256),
            banks={
                'semantic': BankConfig(type='semantic', dim=256),
                'context': BankConfig(type='context', dim=256),
            },
            training=TrainingConfig(batch_size=16, learning_rate=1e-4),
        ),
        'medium-byte': V4Config(
            dim=512,
            backbone=BackboneConfig(dim=512, state_dim=1024, num_layers=12),
            memory=MemoryConfig(dim=512, num_slots=512),  # Smaller memory for speed
            banks={
                'semantic': BankConfig(type='semantic', dim=512),
                'context': BankConfig(type='context', dim=512),
            },
            training=TrainingConfig(batch_size=8, learning_rate=5e-5),
        ),
        'large-byte': V4Config(
            dim=768,
            backbone=BackboneConfig(dim=768, state_dim=1536, num_layers=16),
            memory=MemoryConfig(dim=768, num_slots=1024),
            banks={
                'semantic': BankConfig(type='semantic', dim=768),
                # 'context': BankConfig(type='context', dim=768),
            },
            training=TrainingConfig(batch_size=4, learning_rate=3e-5),
        ),
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Available: {list(configs.keys())}")
    
    return configs[size]
