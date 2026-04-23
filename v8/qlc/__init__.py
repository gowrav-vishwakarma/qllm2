"""Quantum-Logic Core primitives.

- SasakiProjectionMemory (projector.py): rank-r orthonormal subspace with Sasaki updates.
- EffectAlgebraBank (effect_bank.py): bank of M rank-1 Hermitian effects 0 <= E_m <= I.
- OrthoHalt (halt.py): (alpha, beta, gamma) readout from orthocomplement.
- QuantumLogicCore (reason_loop.py): orchestrates probe -> Sasaki update -> halt for T_max iters.
"""

from v8.qlc.projector import SasakiProjectionMemory, SPMState

__all__ = ["SasakiProjectionMemory", "SPMState"]

# Optional re-exports; available once their modules exist.
try:
    from v8.qlc.effect_bank import EffectAlgebraBank  # noqa: F401
    __all__.append("EffectAlgebraBank")
except ImportError:
    pass

try:
    from v8.qlc.halt import OrthoHalt  # noqa: F401
    __all__.append("OrthoHalt")
except ImportError:
    pass

try:
    from v8.qlc.reason_loop import QuantumLogicCore  # noqa: F401
    __all__.append("QuantumLogicCore")
except ImportError:
    pass
