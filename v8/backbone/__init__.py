"""V8 backbone: a fork of the v7 QPAM model + Triton kernels.

Forked from v7/ at the start of the v8 Stage A speed-up effort.
v7/ is intentionally frozen; all backbone-level optimisations land here.
"""

from v8.backbone.model import (
    V7Config,
    V7Block,
    ComplexEmbed,
    ComplexNorm,
    ComplexLinear,
)

__all__ = [
    "V7Config",
    "V7Block",
    "ComplexEmbed",
    "ComplexNorm",
    "ComplexLinear",
]
