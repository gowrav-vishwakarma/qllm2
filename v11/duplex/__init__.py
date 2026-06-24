"""V11 full-duplex speech POC (codec-free, SALMONN-style).

Additive only — does not modify v11.model or shared training code.
"""

from v11.duplex.config import DUPLEX_PRESETS, get_duplex_config, make_duplex_config
from v11.duplex.model import V11DuplexLM

__all__ = [
    "DUPLEX_PRESETS",
    "get_duplex_config",
    "make_duplex_config",
    "V11DuplexLM",
]
