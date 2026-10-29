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
    # N6 operator memory (v14/units.py OperatorMemory). The state is a
    # per-head, per-channel-pair relation inferred from the context and
    # applied to the next input, read into the head next to echo.
    #   'phase' : store unit phasors b * conj(a) / |b * conj(a)| (rotation only;
    #             circular mean), read q * S / (|S| + 1)
    #   'diag'  : store b * conj(a) with per-head RMS-normalised a, b (a plain
    #             diagonal memory: rotation AND magnitude), read q * S
    # '' = off. Needs echo (shares its decode-state slot).
    opmem: str = ''
    opmem_fade_bias: float = -3.9     # retention e^-softplus(-3.9) ~ 0.98 at start
    # Which projection makes the read query q_t from h_t:
    #   'shared' : the key's own W_in (q_t = W_in h_t, a_t = W_in h_{t-1})
    #   'own'    : a separate W_q (B1 run 2: is the shared projection the cause?)
    opmem_query: str = 'shared'
    # Echo with ONE projection for query and key (W_q = W_k), the "full matrix,
    # tied" corner of the B1 run 2 2x2. False = v13 echo (separate W_q, W_k).
    echo_tied_qk: bool = False


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
