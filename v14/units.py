r"""v14 units on the v13_sempty body.

``V14LM`` is the v13_sempty LM; a v14 unit is swapped in behind a
``V14Config`` switch that is off by default, so ``V14LM(V14Config())`` is
bit-identical to the v13 LM (selftest ``test_off_is_v13``).

N7 (``ngram_orders``): in-context n-gram memory, ``v14/ngram.py``. After each
block in ``ngram_layers`` the body gets, per order n, the token that followed
the last earlier occurrence of the current n-gram (exact hash keys, bounded
table, O(1)), through its own zero-init reader. Decode carries one table per
order as the last entry of ``states``.

N9a (``echo_phase_lags``): content-phase echo keys. Echo's query and key are
turned by learned angles from the last few tokens (window ending at t for the
query, at t - 1 for the key), so look-alike template keys ("<noun> is") are
pulled apart by their content instead of colliding (v13 diagnosis). Zero-init:
identical to echo at start. Decode carries (S, h_prev, phase_prev, tail ids).

Removed after a FAIL (code at the listed commit, v14/EXPERIMENT_INDEX.md):
N6 operator memory at ``78e4354``; tied echo at ``5100fd7``; N1 phase
interference between notebooks at ``9e0a3fd``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sempyt.tensor import named

from v13_sempty.model import LM
from v13_sempty.triton_kernels import pam_delta_batched

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
        self.phase_lags = int(getattr(cfg, 'echo_phase_lags', 0) or 0)
        if self.phase_lags:
            if not cfg.echo:
                raise ValueError("N9a echo_phase_lags needs echo")
            planes = cfg.n_heads * cfg.head_dim // 2
            # alpha_t = sum_j W_j E[x_{t-j}]: one angle per head channel pair;
            # zero-init (no RNG drawn), so the model starts as plain echo
            self.phase_proj = nn.Parameter(torch.zeros(self.phase_lags, planes, cfg.dim))
            self.phase_proj._zero_init = True

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

    # ── N9a: content-phase echo keys (raw-torch boundary, like v13's echo) ──

    def _content_phase(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B, L] (-1 = no token) -> angles [B, L, planes]: alpha_t = sum_j W_j E[x_{t-j}]."""
        E = self.embed.embed.weight
        e = E[ids.clamp(min=0)].float()
        e = e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6)     # unit RMS: angles do not track embedding scale
        e = torch.where((ids >= 0)[..., None], e, torch.zeros((), device=e.device))
        alpha = torch.zeros(*ids.shape, self.phase_proj.shape[1], device=ids.device, dtype=torch.float32)
        for j in range(self.phase_lags):
            ej = nn.functional.pad(e, (0, 0, j, 0))[:, :ids.shape[1]]
            alpha = alpha + ej @ self.phase_proj[j].float().t()
        return alpha

    @staticmethod
    def _turn(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """x [B, T, D] turned per channel pair (2i, 2i + 1) by alpha [B, T, D/2]."""
        pairs = x.float().unflatten(-1, (-1, 2))
        c, s = torch.cos(alpha), torch.sin(alpha)
        out = torch.stack([pairs[..., 0] * c - pairs[..., 1] * s,
                           pairs[..., 0] * s + pairs[..., 1] * c], dim=-1)
        return out.flatten(-2).to(x.dtype)

    def _echo_read(self, hidden, input_ids, batch, time, state=None):
        """v13 echo read with N9a content-phase keys (see ``LM._echo_read``)."""
        if not self.phase_lags:
            return super()._echo_read(hidden, input_ids, batch, time, state)
        cfg = self.config
        h = hidden.raw(batch, time, self.model_dim)                     # named-exit: N9a echo boundary
        B, T, D = h.shape
        H, K = cfg.n_heads, cfg.head_dim
        m = self.phase_lags
        q, k = self.echo_qk(h).chunk(2, dim=-1)
        if state is None:
            tail = torch.full((B, m - 1), -1, dtype=torch.long, device=h.device)
            k = nn.functional.pad(k, (0, 0, 1, 0))[:, :T]               # k_t from h_{t-1}; k_0 = 0
        else:
            tail = state[3]
            k_prev = self.echo_qk(state[1].to(h.dtype)).chunk(2, dim=-1)[1]
            k = torch.cat([k_prev[:, None], k[:, :T - 1]], dim=1)
        ids_ext = torch.cat([tail, input_ids.long()], dim=1)            # lags reach into the carried tail
        phi = self._content_phase(ids_ext)[:, m - 1:]                   # [B, T, planes]: window ending at t
        phi_prev = state[2] if state is not None else torch.zeros_like(phi[:, :1, :].squeeze(1))
        phi_key = torch.cat([phi_prev[:, None], phi[:, :T - 1]], dim=1)  # key k_t: window ending at t-1
        q, k = self._turn(q, phi), self._turn(k, phi_key)
        v = nn.functional.embedding(input_ids, self.embed.embed.weight).to(h.dtype)
        v = v * torch.rsqrt(v.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(h.dtype)
        heads = lambda x: x.reshape(B, T, H, K).transpose(1, 2).reshape(B * H, T, K)
        per_head = lambda x: x.transpose(1, 2).reshape(B * H, T)
        dt, wr, er = self.echo_gates(h).split(H, dim=-1)
        retention = per_head(torch.exp(-nn.functional.softplus(dt)))
        bw = per_head(torch.sigmoid(wr))
        be = per_head(torch.sigmoid(er - 3.0).clamp(max=0.95))
        kn = nn.functional.normalize(heads(k), dim=-1)
        if state is None:
            read, S = pam_delta_batched(heads(q), kn, heads(v), retention, bw, be, None)
            S = S.reshape(B * H, K, K).float()
        else:
            S = state[0].reshape(B * H, K, K).float()
            qf, kf, vf = heads(q).float(), kn.float(), heads(v).float()
            g, bwf, bef = retention.float(), bw.float(), be.float()
            reads = []
            for t in range(T):
                kt, vt, qt = kf[:, t], vf[:, t], qf[:, t]
                Sk = torch.einsum('nvk,nk->nv', S, kt)
                erased = S - bef[:, t, None, None] * Sk.unsqueeze(-1) * kt.unsqueeze(-2)
                S = g[:, t, None, None] * erased + bwf[:, t, None, None] * vt.unsqueeze(-1) * kt.unsqueeze(-2)
                reads.append(torch.einsum('nvk,nk->nv', S, qt))
            read = torch.stack(reads, dim=1)
        read = nn.functional.rms_norm(read.float(), (K,), eps=1e-6).to(h.dtype)
        read = read.reshape(B, H, T, K).transpose(1, 2).reshape(B, T, D)
        new_tail = ids_ext[:, ids_ext.shape[1] - (m - 1):] if m > 1 else ids_ext[:, :0]
        new_state = (S.reshape(B, H, K, K), h[:, -1].detach().float(), phi[:, -1].detach(), new_tail)
        return named(self.echo_out(read), (batch, time, self.model_dim)), new_state  # named-exit: echo boundary
