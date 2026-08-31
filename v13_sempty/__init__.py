"""v13_sempty — a simple, pure Phase-Associative Memory (PAM) language model
on the sempyt named-axis frontend.

One notebook per head — a complex d×d matrix — per layer:

    notebook_t = decay_t * notebook_{t-1} + value_t (x) conj(key_t)
    read_t     = d^{-1/2} * (notebook_t . query_t)

Training and prefill run the recurrence in windows via the bounded
log-space retention matrix ``M[s, t] = a_s / a_t = exp(C_t - C_s) <= 1``
(``notebook_s = sum_{t<=s} M[s,t] write_t + a_s notebook_in``); decode runs
the same recurrence one token at a time. The two agree to round-off
(selftest ``test_parallel_vs_recurrent``).

All complex algebra and layout go through sempyt (``NamedTensor``,
``SplitComplex``, ``contract`` / ``outer``); raw torch appears only at the
boundaries declared in ``check_torch_layout.py``, which enforces that
nothing else reaches for ``view`` / ``permute`` / ``[..., 0]``.
"""

from v13_sempty.config import PRESETS, PAMConfig, get_config
from v13_sempty.model import Block, LM, PAMLayer

__all__ = [
    "PAMConfig",
    "LM",
    "PAMLayer",
    "Block",
    "get_config",
    "PRESETS",
]
