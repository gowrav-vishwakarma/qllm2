"""Scale-parametric duplex presets (E3 K=3). Does not touch v11.model.PRESETS."""

import copy
from typing import Dict

from v11.model import V11Config

DUPLEX_VOCAB_SIZE = 512
DUPLEX_TEXT_OFFSET = 8  # special tokens 0..7; text ids >= 8


def make_duplex_config(
    dim: int,
    n_layers: int,
    n_heads: int = 4,
    head_dim: int = 32,
    vocab_size: int = DUPLEX_VOCAB_SIZE,
    n_states: int = 3,
    **kw,
) -> V11Config:
    """Build a V11Config for duplex (locked E3 K=3, head decay, additive write)."""
    return V11Config(
        vocab_size=vocab_size,
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        n_layers=n_layers,
        expand=3,
        dropout=0.1,
        max_seq_len=2048,
        use_learned_pos=False,
        use_rope=True,
        use_gsp=True,
        fused_qkv=True,
        tie_weights=True,
        gradient_checkpointing=False,
        activation='swish',
        chunk_size=128,
        decay_mode='head',
        write_mode='additive',
        n_states=n_states,
        state_dt_spread=2.0,
        base_dt_bias=-4.0,
        **kw,
    )


# Tuned via scripts/count_duplex_params.py (target ~5M / ~10M / ~25M).
DUPLEX_PRESETS: Dict[str, V11Config] = {
    'duplex_5m': make_duplex_config(dim=160, n_layers=8, n_heads=4, head_dim=32),
    'duplex_10m': make_duplex_config(dim=208, n_layers=8, n_heads=6, head_dim=32),
    'duplex_25m': make_duplex_config(dim=304, n_layers=10, n_heads=8, head_dim=32),
}


def get_duplex_config(preset: str = 'duplex_5m') -> V11Config:
    if preset not in DUPLEX_PRESETS:
        raise ValueError(
            f"Unknown duplex preset '{preset}'. Available: {list(DUPLEX_PRESETS.keys())}"
        )
    return copy.deepcopy(DUPLEX_PRESETS[preset])
