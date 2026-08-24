"""V13 selective PAM language model, rewritten on the sempyt named-axis frontend.

Same architecture and parameter names as ``v13`` so a v13 ``state_dict`` loads
directly. Complex algebra and layout bookkeeping go through sempyt
(``NamedTensor``, ``SplitComplex``, ``contract`` / ``outer``); the fused PAM
chunk path and the tied-head CE still drop to torch at ``.raw()`` so the
training math stays numerically equivalent.
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
