"""
V6 Configuration.

Multi-timescale SSM with working memory, external memory layers,
and optional sparse PhaseAttention (disabled by default).
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
    single_bank: bool = False # True = one CGU per layer (no bank pair / coupler)
    dropout: float = 0.1
    max_seq_len: int = 1024

    # SSM output mode
    timescale_separated_output: bool = False  # TSO: separate C_proj per timescale

    # Working memory (0 = disabled; use --wm_slots N to enable)
    num_wm_slots: int = 0    # working memory slots per sequence
    wm_gate_bias: float = -2.0  # start selective (don't write everything)
    wm_read_topk: int = 8     # top-k sparse retrieval (0 = dense softmax)
    wm_slot_decay: float = 0.95  # per-step mask decay for slot freshness

    # Internal memory (0 = disabled; use --im_slots N to enable)
    num_im_slots: int = 0    # internal memory slots (nn.Parameter, trained)
    im_read_topk: int = 8     # top-k sparse retrieval (0 = dense softmax)

    # Episodic memory (event-based, replaces token-wise WM)
    num_episodic_slots: int = 0
    episodic_read_topk: int = 8
    episodic_salience_threshold: float = 0.5

    # External memory flags
    use_persistent_memory: bool = False
    use_session_memory: bool = False
    num_persistent_slots: int = 256
    num_session_slots: int = 128

    # Bank role training
    bank_role_weight: float = 0.05

    # Attention (disabled by default -- model is attention-free unless explicitly enabled)
    use_attention: bool = False
    attn_every: int = 0       # 0 = last layer only; N>0 = every N-th layer
    attn_num_heads: int = 8
    attn_window_size: int = 256

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 20
    warmup_steps: int = 200
    lr_schedule: str = 'warmup_cosine'  # 'cosine' | 'warmup_cosine'
    gradient_clip: float = 1.0
    diversity_loss_weight: float = 0.1
    diversity_loss_floor: float = 0.02
    diversity_margin: float = 0.3

    # Speed / CUDA
    compile_model: bool = False
    compile_mode: str = 'default'     # 'default' | 'reduce-overhead' | 'max-autotune'
    compile_fullgraph: bool = False
    amp_dtype: str = 'auto'           # 'auto' | 'bf16' | 'fp16'
    allow_tf32: bool = True
    num_workers: int = 2
    pin_memory: bool = True

    # Initialization
    init_strategy: str = 'orthogonal'
    init_seed: Optional[int] = None

    # Mode: 'autoregressive' | 'diffusion_text' | 'diffusion_image'
    mode: str = 'autoregressive'

    # Training objective: 'next_token' | 'span_corruption' | 'delayed_recall'
    objective: str = 'next_token'
    span_corruption_rate: float = 0.15
    span_mean_length: int = 3
    delayed_recall_gap: int = 64

    # Diffusion (ignored when mode='autoregressive')
    diffusion_steps: int = 1000
    noise_schedule: str = 'cosine'
    prediction_target: str = 'x0'       # 'x0' | 'epsilon' | 'v'
    diffusion_loss: str = 'mse'         # 'mse' | 'huber'
    sampling_method: str = 'ddpm'       # 'ddpm' | 'ddim'
    ddim_steps: int = 50
    ddim_eta: float = 0.0

    # Image (ignored unless mode='diffusion_image')
    image_size: int = 64
    image_channels: int = 3
    image_encoder: str = 'patch'        # 'patch' | 'fft'
    patch_size: int = 8
    image_dataset: str = 'tiny_imagenet'

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
            batch_size=16, learning_rate=1e-3,
        ),
        'small': V6Config(
            dim=256, state_dim=512, num_layers=8,
            num_banks=2, bank_expand=2,
            batch_size=8, learning_rate=1e-4,
        ),
        'small-matched': V6Config(
            dim=128, state_dim=512, num_layers=12,
            num_banks=2, bank_expand=4,
            batch_size=8, learning_rate=1e-4,
        ),
        'small-rebalanced': V6Config(
            dim=128, state_dim=1280, num_layers=12,
            num_banks=1, bank_expand=4,
            single_bank=True,
            timescale_separated_output=True,
            batch_size=8, learning_rate=1e-4,
        ),
        'medium': V6Config(
            dim=512, state_dim=1024, num_layers=12,
            num_banks=2, bank_expand=2,
            batch_size=4, learning_rate=5e-5,
        ),
        'large': V6Config(
            dim=512, state_dim=1536, num_layers=24,
            num_banks=2, bank_expand=2,
            batch_size=2, learning_rate=3e-5,
            warmup_steps=500,
        ),
        'xl': V6Config(
            dim=768, state_dim=2048, num_layers=32,
            num_banks=2, bank_expand=2,
            batch_size=1, learning_rate=2e-5,
            warmup_steps=1000,
        ),
    }
    if size not in presets:
        raise ValueError(f"Unknown size: {size}. Available: {list(presets.keys())}")
    return presets[size]
