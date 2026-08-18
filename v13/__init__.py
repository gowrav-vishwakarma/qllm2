"""V13: Selective Phase-Associative Memory (PAM) language model.

V11 PAM core with **selective dynamics enabled by default**:

- E2 `write_mode='delta'` — error-correcting erase-before-write
- E3 `n_states=3` — phase-routed multi-state superposition
- Stage-6 `vault_state` + `write_phase_address` — long-term + content-addressable retrieval
- Content-aware GSP gate + gate-surprisal supervision

Identity preserved: complex/phase-first, matrix-state associative memory,
complex-conjugate retrieval. NOT a transformer. O(1)/token inference (no KV cache).
"""

from v13.model import (
    V13Config,
    V13LM,
    V13PAMLayer,
    V13Block,
    get_config,
    PRESETS,
)

__all__ = [
    "V13Config",
    "V13LM",
    "V13PAMLayer",
    "V13Block",
    "get_config",
    "PRESETS",
]
