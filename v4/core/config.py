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
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Configuration for Memory"""
    type: str = "phase_associative"
    num_slots: int = 4096
    dim: int = 256
    params: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class V4Config:
    """
    Full v4 model configuration.
    
    Defines the complete model architecture via the injectable components.
    """
    # Model dimensions
    vocab_size: int = 50257
    dim: int = 256  # Base phase dimension
    max_seq_len: int = 262144  # 256K context
    
    # Phase banks (separate layers)
    banks: Dict[str, BankConfig] = field(default_factory=lambda: {
        'semantic': BankConfig(type='semantic', dim=256),
        'context': BankConfig(type='context', dim=256),
    })
    
    # Coupler
    coupler: CouplerConfig = field(default_factory=CouplerConfig)
    
    # Backbone
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    
    # Memory
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    
    # Objectives
    objectives: List[ObjectiveConfig] = field(default_factory=lambda: [
        ObjectiveConfig(type='ce', weight=1.0),
        ObjectiveConfig(type='coherence', weight=0.01),
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
        
        if 'objectives' in data:
            data['objectives'] = [
                ObjectiveConfig(**obj) if isinstance(obj, dict) else obj
                for obj in data['objectives']
            ]
        
        if 'sampler' in data and isinstance(data['sampler'], dict):
            data['sampler'] = SamplerConfig(**data['sampler'])
        
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        
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
    
    Sizes:
    - 'tiny': For testing (~1M params)
    - 'small': For quick experiments (~10M params)
    - 'medium': Balanced (~50M params)
    - 'large': Production (~200M params)
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
            memory=MemoryConfig(dim=256, num_slots=2048),
            banks={
                'semantic': BankConfig(type='semantic', dim=256),
                'context': BankConfig(type='context', dim=256),
            },
            training=TrainingConfig(batch_size=8, learning_rate=1e-4),
        ),
        'medium': V4Config(
            dim=512,
            backbone=BackboneConfig(dim=512, state_dim=1024, num_layers=12),
            memory=MemoryConfig(dim=512, num_slots=4096),
            banks={
                'semantic': BankConfig(type='semantic', dim=512),
                'context': BankConfig(type='context', dim=512),
                'language': BankConfig(type='language', dim=512),
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
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Available: {list(configs.keys())}")
    
    return configs[size]
