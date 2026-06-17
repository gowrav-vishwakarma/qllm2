"""V11: Matured Phase-Associative Memory (PAM) language model.

Builds on the proven V7 core (complex CGU + PAM + RoPE + GSP + ModSwish,
chunked dual-form training, O(1) recurrent inference) and adds genuinely new,
in-family **memory dynamics** behind single-variable config flags:

- E1 `decay_mode='per_channel'`  -- per-key-channel data-dependent decay.
- E2 `write_mode='delta'`        -- delta-rule (error-correcting) state write.
- E3 `n_states>1`               -- multi-state superposed PAM with phase routing.

Defaults (`decay_mode='head'`, `write_mode='additive'`, `n_states=1`) reproduce
the V7 7d baseline exactly, so every lever is a clean ablation.

Identity preserved: complex/phase-first, matrix-state associative memory,
complex-conjugate retrieval. NOT a transformer, NOT a vector-state SSM.
Inference stays O(1)/token with bounded state (no KV cache).
"""

from v11.model import (
    V11Config,
    V11LM,
    V11PAMLayer,
    V11Block,
    get_config,
    PRESETS,
)

__all__ = [
    "V11Config",
    "V11LM",
    "V11PAMLayer",
    "V11Block",
    "get_config",
    "PRESETS",
]
