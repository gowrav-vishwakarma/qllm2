"""V14Config: the v13_sempty PAMConfig plus the v14 unit switches.

Every switch is off by default, so ``V14Config()`` is exactly the v13_sempty
model. Presets build on v13_sempty's.
"""

import copy
from dataclasses import dataclass, fields

from v13_sempty.config import PAMConfig, get_config as v13_get_config


@dataclass
class V14Config(PAMConfig):
    # N1 phase interference (v11 E3 on the lean body). K delta notebooks per
    # head, each with its own decay offset; their reads are combined by
    #   'sum'   : y = sum_k a_{h,k} y_k              (learned static mix)
    #   'gate'  : y = 1/K sum_k (1 + c_k(x)) y_k     (real data-dependent mix)
    #   'phase' : y = 1/K sum_k R(phi_k(x)) y_k       (rotation interference)
    # phi, c = W x (zero-init), so all three equal the plain average at start.
    # Needs delta (the notebooks are delta notebooks). 1 = off.
    phase_states: int = 1
    phase_read: str = 'phase'
    phase_dt_spread: float = 2.0


def from_pam(cfg: PAMConfig, **overrides) -> V14Config:
    """Lift a v13_sempty PAMConfig into a V14Config (v14 switches off)."""
    out = V14Config(**{f.name: copy.deepcopy(getattr(cfg, f.name)) for f in fields(PAMConfig)})
    for key, value in overrides.items():
        if not hasattr(out, key):
            raise AttributeError(f"V14Config has no field '{key}'")
        setattr(out, key, value)
    return out


def get_config(preset: str = 'baseline_real_pm', **overrides) -> V14Config:
    return from_pam(v13_get_config(preset), **overrides)
