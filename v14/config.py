"""V14Config: the v13_sempty PAMConfig plus the v14 unit switches.

Every switch is off by default, so ``V14Config()`` is exactly the v13_sempty
model. Presets build on v13_sempty's.
"""

import copy
from dataclasses import dataclass, fields

from v13_sempty.config import PAMConfig, get_config as v13_get_config


@dataclass
class V14Config(PAMConfig):
    """No v14 switch is live (N6, tied echo and N1 were removed after a FAIL)."""


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
