r"""v14 units on the v13_sempty body.

``V14LM`` is the v13_sempty LM; a new v14 unit is swapped in here behind a
``V14Config`` switch that is off by default, so ``V14LM(V14Config())`` is
bit-identical to the v13 LM (selftest ``test_off_is_v13``).

No unit is live right now. Removed after a FAIL (code at the listed commit,
v14/EXPERIMENT_INDEX.md): N6 operator memory at ``78e4354``; tied echo at
``5100fd7``; N1 phase interference between notebooks at ``9e0a3fd``.
"""

from __future__ import annotations

from v13_sempty.model import LM

from v14.config import V14Config


class V14LM(LM):
    """The v13_sempty LM with v14 units swapped in. ``V14Config()`` == v13 LM."""

    def __init__(self, cfg: V14Config):
        super().__init__(cfg)
