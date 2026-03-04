"""
V5 Configuration.

Simplified from V4: no registry system, no YAML, just dataclasses.
Provides presets that match the reviewer's benchmark setup.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class V5Config:
    # Model
    vocab_size: int = 50257
    dim: int = 256            # complex dim (= 512 real values per position)
    state_dim: int = 512      # SSM hidden state dimension
    num_layers: int = 8
    num_banks: int = 2        # number of algebraic banks per layer
    bank_expand: int = 2      # CGU expansion factor
    num_heads: int = 8        # attention heads (for PhaseAttention layers)
    attn_every_k: int = 4     # place PhaseAttention every K layers (0=none)
    window_size: int = 256    # sliding window size for attention
    dropout: float = 0.1
    max_seq_len: int = 1024

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 20
    warmup_steps: int = 200
    gradient_clip: float = 1.0
    diversity_loss_weight: float = 0.05

    # Speed
    compile_model: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'V5Config':
        with open(path) as f:
            return cls(**json.load(f))


def get_config(size: str = 'small') -> V5Config:
    """
    Preset configs. 'small' matches the reviewer's benchmark setup:
    256 dim, 8 layers, trained on 20k TinyStories for 20 epochs.
    """
    presets = {
        'tiny': V5Config(
            dim=64, state_dim=128, num_layers=4,
            num_banks=2, num_heads=4, bank_expand=2,
            batch_size=16, learning_rate=1e-3,
        ),
        'small': V5Config(
            dim=256, state_dim=512, num_layers=8,
            num_banks=2, num_heads=8, bank_expand=2,
            batch_size=8, learning_rate=1e-4,
        ),
        # Matched to reviewer's ~8M real params: complex dim 128 = 256 real
        'small-matched': V5Config(
            dim=128, state_dim=256, num_layers=8,
            num_banks=2, num_heads=4, bank_expand=2,
            batch_size=8, learning_rate=1e-4,
        ),
        'medium': V5Config(
            dim=512, state_dim=1024, num_layers=12,
            num_banks=3, num_heads=8, bank_expand=2,
            batch_size=4, learning_rate=5e-5,
        ),
    }
    if size not in presets:
        raise ValueError(f"Unknown size: {size}. Available: {list(presets.keys())}")
    return presets[size]
