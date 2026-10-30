"""Build v14 units from the v13_sempty trainer and checkpoint loaders.

The v13 session trains and probes 600M models; v14 units live only here
(v14/units.py), so v13 builds them through this hook instead of keeping a
second copy. A v13 config with every v14 switch off still builds the plain
v13 LM, bit for bit.

    trainer   cfg = with_v14_switches(cfg, echo_phase_lags=3); model = build_lm(cfg)
    loaders   cfg = config_from_raw(payload['config']); model = build_lm(cfg)
"""

from __future__ import annotations

from dataclasses import fields

from v13_sempty.config import PAMConfig

# A checkpoint config is a v14 one when any of these is on.
V14_SWITCHES = ('echo_phase_lags', 'ngram_orders', 'opmem')


def is_v14(raw) -> bool:
    """True if a config (dict or dataclass) turns on any v14 unit."""
    get = raw.get if isinstance(raw, dict) else (lambda k, d=None: getattr(raw, k, d))
    return any(get(k) for k in V14_SWITCHES)


def with_v14_switches(cfg: PAMConfig, **switches):
    """A v13 PAMConfig plus v14 switches -> V14Config (unchanged if all off)."""
    if not any(switches.values()):
        return cfg
    from v14.config import from_pam
    return from_pam(cfg, **switches)


def config_from_raw(raw: dict):
    """A checkpoint's saved config dict -> PAMConfig, or V14Config if it uses v14 units."""
    if is_v14(raw):
        from v14.config import V14Config
        names = {f.name for f in fields(V14Config)}
        kw = {k: v for k, v in raw.items() if k in names}
        for k in ('ngram_orders', 'ngram_layers', 'pam_layers'):
            if k in kw and isinstance(kw[k], list):
                kw[k] = tuple(kw[k])
        return V14Config(**kw)
    names = set(PAMConfig.__dataclass_fields__)
    return PAMConfig(**{k: v for k, v in raw.items() if k in names})


def build_lm(cfg):
    """V14LM for a config with a v14 unit on, else the v13 LM."""
    if is_v14(cfg):
        from v14.units import V14LM
        return V14LM(cfg)
    from v13_sempty.model import LM
    return LM(cfg)
