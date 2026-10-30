r"""v14 units on the v13_sempty body.

``V14LM`` is the v13_sempty LM; a v14 unit is swapped in behind a
``V14Config`` switch that is off by default, so ``V14LM(V14Config())`` is
bit-identical to the v13 LM (selftest ``test_off_is_v13``).

N7 (``ngram_orders``): in-context n-gram memory, ``v14/ngram.py``. After each
block in ``ngram_layers`` the body gets, per order n, the token that followed
the last earlier occurrence of the current n-gram (exact hash keys, bounded
table, O(1)), through its own zero-init reader. Decode carries one table per
order as the last entry of ``states``.

Removed after a FAIL (code at the listed commit, v14/EXPERIMENT_INDEX.md):
N6 operator memory at ``78e4354``; tied echo at ``5100fd7``; N1 phase
interference between notebooks at ``9e0a3fd``.
"""

from __future__ import annotations

import torch.nn as nn

from sempyt.tensor import named

from v13_sempty.model import LM

from v14.config import V14Config
from v14.ngram import NGramReader, lookup_orders


class V14LM(LM):
    """The v13_sempty LM with v14 units swapped in. ``V14Config()`` == v13 LM."""

    def __init__(self, cfg: V14Config):
        super().__init__(cfg)
        self.ngram_orders = tuple(getattr(cfg, 'ngram_orders', ()) or ())
        self.ngram_layers = ()
        if self.ngram_orders:
            if cfg.is_complex:
                raise ValueError("N7 is implemented for the real arm")
            if cfg.share_reads:
                raise ValueError("N7 needs its own reader per block: M2 share_reads is not supported")
            L = cfg.n_layers
            self.ngram_layers = tuple(sorted({i % L for i in cfg.ngram_layers}))
            # zero-init readers, created after the body's init: no RNG drawn,
            # so the rest of the model initialises exactly as without N7
            self.ngram_read = nn.ModuleDict({str(i): NGramReader(cfg.dim, self.ngram_orders)
                                             for i in self.ngram_layers})
        self._ng_in, self._ng_out, self._ng_want_state = None, None, False

    def _run_blocks(self, z, batch, time, states, step_offset, input_ids=None):
        if not self.ngram_orders:
            return super()._run_blocks(z, batch, time, states, step_offset, input_ids)
        if self.config.gradient_checkpointing and self.training and states is None:
            raise NotImplementedError("N7: gradient checkpointing not wired yet")
        value, found, self._ng_out = lookup_orders(
            input_ids, self.ngram_orders, self.config.ngram_slots,
            state=self._ng_in if states is not None else None, want_state=self._ng_want_state)
        embedded = self.embed.embed.weight[value]                       # [k, B, T, D]
        new_states = []
        for i, block in enumerate(self.blocks):
            state = states[i] if states is not None else None
            z, new_state = block(z, pam_state=state, step_offset=step_offset)
            new_states.append(new_state)
            if self.cond_mem_layers and i in self.cond_mem_layers:
                z = self.cond_mem[str(i)](z, input_ids)
            if i in self.ngram_layers:
                h = z.raw(batch, time, self.model_dim)                   # named-exit: N7 boundary
                r = self.ngram_read[str(i)](h, embedded, found)
                z = named(h + r, (batch, time, self.model_dim))          # named-exit: N7 boundary
        return z, new_states

    def forward(self, input_ids, states=None, step_offset: int = 0, labels=None):
        if not self.ngram_orders:
            return super().forward(input_ids, states, step_offset, labels)
        ng = None
        if states is not None:
            states, ng = list(states[:-1]), states[-1]
        self._ng_in, self._ng_want_state = ng, True
        try:
            logits, new_states, aux = super().forward(input_ids, states, step_offset, labels)
        finally:
            self._ng_in, self._ng_want_state = None, False
        return logits, list(new_states) + [self._ng_out], aux
