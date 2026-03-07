"""
V6 Configuration.

No attention. Multi-timescale SSM with working memory and external memory layers.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class V6Config:
    # Model
    vocab_size: int = 50257
    dim: int = 128            # complex dim (= 256 real values per position)
    state_dim: int = 512      # SSM hidden state dimension
    num_layers: int = 12
    num_banks: int = 2        # named banks: semantic + context
    bank_expand: int = 4      # CGU expansion factor
    dropout: float = 0.1
    max_seq_len: int = 1024

    # Working memory
    num_wm_slots: int = 64    # working memory slots per sequence
    wm_gate_bias: float = -2.0  # start selective (don't write everything)

    # Internal memory
    num_im_slots: int = 128   # internal memory slots (nn.Parameter, trained)

    # External memory flags
    use_persistent_memory: bool = False  # persistent memory (per-user, cross-session)
    use_session_memory: bool = False     # session memory (optional, disabled by default)
    num_persistent_slots: int = 256
    num_session_slots: int = 128

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

    # Initialization
    init_strategy: str = 'orthogonal'
    init_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'V6Config':
        with open(path) as f:
            return cls(**json.load(f))


def get_config(size: str = 'small-matched') -> V6Config:
    """Preset configs for V6."""
    presets = {
        'tiny': V6Config(
            dim=64, state_dim=128, num_layers=4,
            num_banks=2, bank_expand=2,
            num_wm_slots=16, num_im_slots=32,
            batch_size=16, learning_rate=1e-3,
        ),
        'small': V6Config(
            dim=256, state_dim=512, num_layers=8,
            num_banks=2, bank_expand=2,
            num_wm_slots=64, num_im_slots=128,
            batch_size=8, learning_rate=1e-4,
        ),
        'small-matched': V6Config(
            dim=128, state_dim=512, num_layers=12,
            num_banks=2, bank_expand=4,
            num_wm_slots=64, num_im_slots=128,
            batch_size=8, learning_rate=1e-4,
        ),
        'medium': V6Config(
            dim=512, state_dim=1024, num_layers=12,
            num_banks=2, bank_expand=2,
            num_wm_slots=128, num_im_slots=256,
            batch_size=4, learning_rate=5e-5,
        ),
    }
    if size not in presets:
        raise ValueError(f"Unknown size: {size}. Available: {list(presets.keys())}")
    return presets[size]
