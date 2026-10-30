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

from sempyt import exp, mean, scan, select, shift, sigmoid, softplus, sqrt, square
from sempyt import cat, stack, take
from sempyt.dim import Dim, ProductDim
from sempyt.nn import Linear as NamedLinear
from sempyt.policies import REAL, SplitComplex
from sempyt.tensor import NamedTensor, named

from v13_sempty.model import LM
from v13_sempty.triton_kernels import pam_delta_batched

from v14.config import V14Config
from v14.ngram import NGramReader, lookup_orders

OPMEM_QUERIES = ('shared', 'own')
OPMEM_MODES = ('phase', 'diag', 'coord')


class OperatorMemory(nn.Module):
    r"""N6 / N10: a memory of how things change, applied to the next input.

    N10 (``coord``, user idea "rules as phase continuation"): a and b are not
    free complex vectors but ONE learned coordinate per head, read on P
    circles of geometrically spread periods (grid-cell / residue code), so a
    rule is one step along the coordinate, the same step on every circle.

    Each channel pair of each head is one complex number. At every token t:

        a_t = W_in h_{t-1}            the input side   (what came before)
        b_t = W_role ê(x_t)           the output side  (the token that followed)
        r_t = b_t · conj(a_t)         their relation: a rotation (and a scale)
        S_t = γ_t S_{t-1} + w_t r_t   relations accumulated, faded per head
        q_t = W_in h_t                the next input (W_q h_t with opmem_query='own')
        p_t = q_t · S_t / (|S_t| + 1)  (phase)   or   q_t · S_t  (diag)

    ``phase`` keeps only the rotation of each relation (r / |r|), so S is a
    weighted circular mean and the read TURNS the next input by the mean
    turn: d numbers per head, inferable from one or two examples, composing
    by adding angles. ``diag`` keeps rotation and magnitude, a plain diagonal
    memory with the same parameters: the control for the "rotation only" bias.
    ê is the unit-RMS embedding of the current token; w (write) and γ (fade)
    are per-head gates read from h_t. The read is RMS-normalised per head and
    added to the pre-logit vector through a zero-init projection.
    """

    def __init__(self, cfg: V14Config, model_dim: Dim):
        super().__init__()
        if cfg.opmem not in OPMEM_MODES:
            raise ValueError(f"opmem must be one of {OPMEM_MODES}")
        if cfg.opmem_query not in OPMEM_QUERIES:
            raise ValueError(f"opmem_query must be one of {OPMEM_QUERIES}")
        self.mode = cfg.opmem
        self.model_dim = model_dim
        self.heads = Dim("op_heads", cfg.n_heads)
        self.plane = Dim("op_plane", cfg.head_dim // 2)
        self.pair = Dim("op_pair", 2)
        self.feat = ProductDim(self.heads, self.plane, self.pair)
        own = cfg.opmem_query == 'own'
        if self.mode == 'coord':
            # N10: one learned coordinate per head, read on P circles whose
            # periods spread geometrically (angle_p = omega_p * coordinate)
            self.w_in = NamedLinear(model_dim, self.heads, bias=False)
            self.w_role = NamedLinear(model_dim, self.heads, bias=False)
            self.w_q = NamedLinear(model_dim, self.heads, bias=False) if own else None
            P = cfg.head_dim // 2
            self.register_buffer('omega', cfg.opmem_coord_base ** (-torch.arange(P, dtype=torch.float32) / P),
                                 persistent=False)
        else:
            self.w_in = NamedLinear(model_dim, self.feat, bias=False)
            self.w_role = NamedLinear(model_dim, self.feat, bias=False)
            # The query's projection: the key's W_in ('shared') or its own W_q ('own').
            self.w_q = NamedLinear(model_dim, self.feat, bias=False) if own else None
        self.write = NamedLinear(model_dim, self.heads)
        self.write._zero_init = True                      # w = 1/2 at start
        self.fade = NamedLinear(model_dim, self.heads)
        self.fade._zero_init = True
        self.fade._init_bias = cfg.opmem_fade_bias
        self.out = NamedLinear(self.feat, model_dim, bias=False)
        self.out._zero_init = True                        # identity at start
        self.eps = 1e-6

    def _complex(self, x: NamedTensor, batch: Dim, time: Dim) -> NamedTensor:
        return x.to(batch, time, self.heads, self.plane, self.pair).with_policy(
            SplitComplex(self.pair))

    def _circles(self, coord: NamedTensor, batch: Dim, time: Dim) -> NamedTensor:
        """N10 (raw-torch boundary): coordinate (batch, time, heads) -> unit complex
        e^{i omega_p coord} on every circle p: (batch, time, heads, plane | complex)."""
        c = coord.raw(batch, time, self.heads).float()                   # named-exit: N10 circles
        ang = c[..., None] * self.omega
        z = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
        return named(z, (batch, time, self.heads, self.plane, self.pair)).with_policy(
            SplitComplex(self.pair))                                      # named-exit: N10 circles

    def _head_rms(self, z: NamedTensor) -> NamedTensor:
        """sqrt(mean |z|² over the head's planes) + eps, per head."""
        return sqrt(mean(square(z.abs()), over=self.plane) + self.eps)

    def forward(self, hidden: NamedTensor, embedded: NamedTensor, state=None):
        """hidden, embedded: (batch, time, model_dim). state = (h_last, S_last) or None.
        Returns the pre-logit contribution (batch, time, model_dim) and the new state."""
        batch, time = hidden.layout[0], hidden.layout[1]
        p, new_state = self.read(hidden, embedded, state)
        return self.out(p.with_policy(REAL).to(batch, time, self.feat)), new_state

    def read(self, hidden: NamedTensor, embedded: NamedTensor, state=None):
        """The operator read p_t (batch, time, heads, plane | complex), RMS-normalised per head."""
        batch, time = hidden.layout[0], hidden.layout[1]
        T = time.size
        if state is None:
            h_prev = shift(hidden, time, 1)
        else:
            last = self._restate(state[0], (batch, self.model_dim))
            if T == 1:
                h_prev = stack([last], into=time, at=self.model_dim)
            else:
                first = stack([last], into=Dim("time", 1), at=self.model_dim)
                h_prev = cat([first, take(hidden, time, 0, T - 1)], over=time, into=time)
        unit_embed = embedded / sqrt(mean(square(embedded), over=self.model_dim) + self.eps)
        w_q = self.w_in if self.w_q is None else self.w_q
        if self.mode == 'coord':
            # r = e^{i omega (coord_b - coord_a)}: the step from input to output
            a = self._circles(self.w_in(h_prev), batch, time)
            b = self._circles(self.w_role(unit_embed), batch, time)
            q = self._circles(w_q(hidden), batch, time)
        else:
            a = self._complex(self.w_in(h_prev), batch, time)
            b = self._complex(self.w_role(unit_embed), batch, time)
            q = self._complex(w_q(hidden), batch, time)
        if self.mode in ('phase', 'coord'):
            r = b * a.conj()
            r = r / (r.abs() + self.eps)
        else:
            r = (b / self._head_rms(b)) * (a / self._head_rms(a)).conj()
        w = sigmoid(self.write(hidden))                                  # (batch, time, heads)
        g = exp(-softplus(self.fade(hidden)))                            # retention in (0, 1)
        # scan returns time last; split-complex ops (abs, conj, cmul) need the
        # pair axis last, so restore that order.
        S = scan(r * w, g, over=time, mode='chunked').to(
            batch, time, self.heads, self.plane, self.pair)
        if state is not None:
            S_in = self._restate(state[1], (batch, self.heads, self.plane, self.pair)
                                 ).with_policy(SplitComplex(self.pair))
            S = S + exp(cumsum_log(g, time)) * S_in
        if self.mode in ('phase', 'coord'):
            p = q * S / (S.abs() + 1.0)
        else:
            p = q * S
        p = p / self._head_rms(p)
        new_state = (select(hidden, time, T - 1).data.detach(),          # named-exit: decode state
                     select(S, time, T - 1).to(batch, self.heads, self.plane, self.pair).data)  # named-exit: decode state
        return p, new_state

    @staticmethod
    def _restate(data, layout):
        """A carried raw state back onto this call's axes (no data moves)."""
        return named(data, layout)  # named-exit: restate


def cumsum_log(g: NamedTensor, over: Dim) -> NamedTensor:
    """log of the running product of g along ``over``: Σ_{k≤t} log g_k."""
    from sempyt import cumsum, log
    return cumsum(log(g), over=over)


def cumsum_log(g: NamedTensor, over: Dim) -> NamedTensor:
    """log of the running product of g along ``over``: Σ_{k≤t} log g_k."""
    from sempyt import cumsum, log
    return cumsum(log(g), over=over)




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
        self.opmem = None
        if getattr(cfg, 'opmem', ''):
            if cfg.is_complex or not cfg.echo:
                raise ValueError("N6/N10 opmem is implemented on the real echo body")
            # Build and init the new memory WITHOUT moving the body: the body's
            # params and the global RNG stay exactly as plain echo at this seed
            # (paired formation tests need identical starts).
            body = {k: v.clone() for k, v in self.state_dict().items()}
            with torch.random.fork_rng(devices=[]):
                self.opmem = OperatorMemory(cfg, self.model_dim)
                self._init_weights()
            missing = self.load_state_dict(body, strict=False)
            assert all(k.startswith('opmem.') for k in missing.missing_keys), missing
            if self.echo:
                with torch.no_grad():
                    self.echo_gates.bias[:cfg.n_heads].fill_(cfg.echo_dt_bias)
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
        use_ckpt = self.config.gradient_checkpointing and self.training and states is None
        value, found, self._ng_out = lookup_orders(
            input_ids, self.ngram_orders, self.config.ngram_slots,
            state=self._ng_in if states is not None else None, want_state=self._ng_want_state)
        embedded = self.embed.embed.weight[value]                       # [k, B, T, D]
        new_states = []
        for i, block in enumerate(self.blocks):
            state = states[i] if states is not None else None
            if use_ckpt:                                   # as LM._run_blocks: recompute in backward
                z, new_state = self._ckpt_block(block, z, step_offset)
            else:
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

    # ── N6 / N10 operator memory on the pre-logit vector ─────────────────────

    def _lm_vector(self, z, input_ids, batch, time, echo_state=None):
        """Head + echo (as in v13) + operator memory. With opmem the echo state
        slot carries (echo_state, op_state)."""
        if self.opmem is None:
            return super()._lm_vector(z, input_ids, batch, time, echo_state)
        e_state, o_state = echo_state if echo_state is not None else (None, None)
        hidden = self.output_norm(z)
        lm = self.lm_head_norm(self.lm_head_proj(hidden))
        out, new_echo = self._echo_read(hidden, input_ids, batch, time, e_state)
        lm = lm + out
        op_out, new_op = self.opmem(hidden, self.embed(input_ids, batch, time), o_state)
        return lm + op_out, (new_echo, new_op)
