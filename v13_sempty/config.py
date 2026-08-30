"""PAMConfig + presets for the simple PAM.

Only the knobs the model actually has. Everything v13 learned around the
recurrence (E1/E2/E3, GSP, vault, gate supervision) is gone by design —
see EXPERIMENTS_SEMPY.md for the decision log.
"""

import copy
from dataclasses import dataclass


@dataclass
class PAMConfig:
    vocab_size: int = 50257
    dim: int = 384
    n_heads: int = 6
    head_dim: int = 64
    n_layers: int = 16
    expand: int = 3
    dropout: float = 0.1
    max_seq_len: int = 2048
    use_rope: bool = True
    tie_weights: bool = True
    gradient_checkpointing: bool = True
    activation: str = 'swish'        # 'swish' (v7/v11 default) | 'modrelu' | 'phase_mod'
    chunk_size: int = 256            # notebook carry window (train / prefill)
    base_dt_bias: float = -4.0       # decay bias: init decay ~= e^-4 ~= 0.018
    is_complex: bool = True          # SplitComplex (phase) | fully-real PAM


def _base_flat(**kw) -> PAMConfig:
    cfg = PAMConfig(
        vocab_size=50257, dim=384, n_heads=6, head_dim=64,
        n_layers=16, expand=3, dropout=0.1, max_seq_len=2048,
        activation='swish', chunk_size=256,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


PRESETS = {
    # The production shape: v11 hero's bare minimum (7d baseline geometry).
    'baseline': _base_flat(),
    # The fully-real twin of baseline at matched real width: 768 real channels
    # (== 384 complex), 128-dim real heads. Same recurrence, real arithmetic.
    'baseline_real': _base_flat(dim=768, head_dim=128, is_complex=False),
    # Small enough to run on CPU in a minute.
    'micro': PAMConfig(
        vocab_size=50261, dim=96, n_heads=3, head_dim=32, n_layers=6,
        expand=3, dropout=0.0, max_seq_len=2048, chunk_size=64,
        gradient_checkpointing=False,
    ),
    # The dev preset: tiny, fast, for selftest / smoke runs.
    'tiny': PAMConfig(
        vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False,
    ),
    # The dev twin of tiny: fully-real, CPU-friendly selftest preset.
    'tiny_real': PAMConfig(
        vocab_size=50257, dim=128, n_heads=2, head_dim=64, n_layers=2,
        expand=2, dropout=0.0, max_seq_len=512, chunk_size=64,
        gradient_checkpointing=False, is_complex=False,
    ),
}


def get_config(preset: str = 'baseline') -> PAMConfig:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
    return copy.deepcopy(PRESETS[preset])
