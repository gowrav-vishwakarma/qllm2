"""V13 selective PAM language model, rewritten on the sempyt named-axis frontend.

Same architecture and parameter names as ``v13`` so a v13 ``state_dict`` loads
directly, and numerically equivalent to it (see ``selftest.py``). Complex
algebra and all layout bookkeeping go through sempyt (``NamedTensor``,
``SplitComplex``, ``contract`` / ``take`` / ``solve_triangular``); raw torch
appears only at the boundaries listed in ``SEMPYT_OPS.md``, which
``check_torch_layout.py`` enforces.

Only the production path is implemented: E3 (K notebooks) with the delta write
rule and head-scalar decay — chunked-parallel for training and prefill, O(1)
recurrent for decode.
"""

from v13_sempty.config import PRESETS, V13Config, get_config
from v13_sempty.model import V13Block, V13LM, V13PAMLayer

__all__ = [
    "V13Config",
    "V13LM",
    "V13PAMLayer",
    "V13Block",
    "get_config",
    "PRESETS",
]
