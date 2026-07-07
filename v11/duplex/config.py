"""Scale-parametric duplex presets (E3 K=3). Does not touch v11.model.PRESETS."""

import copy
from typing import Dict

from v11.model import V11Config

DUPLEX_VOCAB_SIZE = 512
DUPLEX_TEXT_OFFSET = 8  # special tokens 0..7; text ids >= 8 (legacy 5M/10M/25M POC)

# Unified-vocab default for the real voice model (see v11/duplex/tokenizer.py):
# 16 control + 32000 text (SentencePiece) + 4*2048 codec (Mimi) = 40208.
DUPLEX_100M_VOCAB = 16 + 32000 + 4 * 2048


def make_duplex_config(
    dim: int,
    n_layers: int,
    n_heads: int = 4,
    head_dim: int = 32,
    vocab_size: int = DUPLEX_VOCAB_SIZE,
    n_states: int = 3,
    chunk_size: int = 128,
    gate_content_aware: bool = False,
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
        chunk_size=chunk_size,
        decay_mode='head',
        write_mode='additive',
        n_states=n_states,
        state_dt_spread=2.0,
        base_dt_bias=-4.0,
        gate_content_aware=gate_content_aware,
        **kw,
    )


# Tuned via scripts/count_duplex_params.py (target ~5M / ~10M / ~25M).
DUPLEX_PRESETS: Dict[str, V11Config] = {
    'duplex_5m': make_duplex_config(dim=160, n_layers=8, n_heads=4, head_dim=32),
    'duplex_10m': make_duplex_config(dim=208, n_layers=8, n_heads=6, head_dim=32),
    'duplex_25m': make_duplex_config(dim=304, n_layers=10, n_heads=8, head_dim=32),
    # Production voice model: proven v11_e3_k3 geometry (16x384, K=3, phase-aware
    # gate), unified ~40k vocab. Trains at B<=8/seq2048 on a 24GB RTX 4090.
    'duplex_100m': make_duplex_config(
        dim=384, n_layers=16, n_heads=6, head_dim=64,
        vocab_size=DUPLEX_100M_VOCAB, chunk_size=256, gate_content_aware=True,
    ),
}


def get_duplex_config(preset: str = 'duplex_5m', vocab_size: int = 0) -> V11Config:
    if preset not in DUPLEX_PRESETS:
        raise ValueError(
            f"Unknown duplex preset '{preset}'. Available: {list(DUPLEX_PRESETS.keys())}"
        )
    cfg = copy.deepcopy(DUPLEX_PRESETS[preset])
    if vocab_size > 0:  # override with the tokenizer's actual total vocab
        cfg.vocab_size = vocab_size
    return cfg
