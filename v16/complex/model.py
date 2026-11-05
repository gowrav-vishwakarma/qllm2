"""Complex PAM with the paper body's parts (cx_m2; user 2026-11-05: "keep all complex things in v16/complex").

The v13 complex ``PAMLayer`` / ``Block`` carry only the plain notebook. These subclasses add, each identity / off at init like its
real twin in ``v13_sempty/model.py``:
- A3 delta erase/write: unit complex keys, S_t = g_t S (I - b_e k k^H) + b_w v k^H, y = S q. The prefill runs the REAL fused
  ``pam_delta_batched`` on the realified notebook: for a unit complex key the projector k k^H realifies to two ORTHONORMAL rank-1
  projectors (r(k), J r(k); J = multiplication by i), so one complex step = two real sub-steps (decay on the first). Decode is an
  independent true-complex stepwise reference.
- N1 Chrono: the real twin's per-head warp on the complex RoPE angle (same inv_freq), exclusive cumulative clock carried in state.
- N4 read-out gate: zero-init silu gate, exactly 1 at start.
- RESET_AT_EOS: the keep mask zeroes retention at document separators (delta path).
- M1 sparse notebooks (pam_layers) + an explicit gated-MLP width (mlp_hidden) for param matching.
- M2 shared reads: a complex reader of the writer's realified WY write, query rotated by the writer's clock; zero-init out-proj.
- E1 echo: the succession memory at the head in complex form (keys from h_{t-1}, values = the RMS-unit complex token embedding,
  the same realified delta kernel, zero-init complex out-proj).
``v14/hook.py:build_lm`` routes a complex config with any of these on to ``CxLM``.
Parity: ``v16/complex/tests/cx_parity.py``.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from sempyt.dim import Dim
from sempyt.nn import Linear as NamedLinear
from sempyt.policies import SplitComplex
from sempyt.tensor import NamedTensor, named
from v13_sempty.config import PAMConfig
from v13_sempty.complex_ops import (ComplexGatedUnit, ComplexLinear, ComplexNorm, _build_activation,
                                    as_complex_dropout_mask, to_real_concat)
from v13_sempty.model import LM, PAMLayer
from v13_sempty.triton_kernels import pam_delta_batched, pam_shared_read

PORTED = ('delta', 'chrono', 'out_gate', 'pam_layers', 'share_reads', 'mlp_hidden', 'echo')


def wants_cx(cfg) -> bool:
    """A complex config that uses any ported part (routes build_lm to CxLM)."""
    return bool(getattr(cfg, 'is_complex', False)) and any(getattr(cfg, k, None) for k in PORTED)


# ── realified complex delta, shared by the notebooks' twin (CxPAMLayer has its own copy) and E1 echo ─────────────────────────────

def _cx_J(x):                                     # [..., K, 2] -> i * x
    return torch.stack((-x[..., 1], x[..., 0]), dim=-1)


def _cx_unit(k):                                  # unit complex vectors over (K, pair)
    n = k.float().pow(2).sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12).sqrt()
    return (k.float() / n).to(k.dtype)


def _realify(S):                                  # complex [..., K, K] -> realified [..., 2K, 2K]
    *lead, K, _ = S.shape
    R = torch.zeros(*lead, 2 * K, 2 * K, dtype=torch.float32, device=S.device)
    R[..., 0::2, 0::2] = S.real; R[..., 0::2, 1::2] = -S.imag
    R[..., 1::2, 0::2] = S.imag; R[..., 1::2, 1::2] = S.real
    return R


def _complexify(R):                               # realified [..., 2K, 2K] -> complex [..., K, K]
    return torch.complex(R[..., 0::2, 0::2].float(), R[..., 1::2, 0::2].float())


def cx_delta_chunked(q, k, v, g, bw, be, carry=None):
    """q, k (unit), v [N, T, K, 2]; g, bw, be [N, T]; carry realified [N, 2K, 2K] or None -> (read [N, T, K, 2], S realified)."""
    N, T, K, _ = q.shape
    two = lambda a, b: torch.stack((a, b), dim=2).reshape(N, 2 * T, *a.shape[2:])
    r = lambda x: x.reshape(N, T, 2 * K)
    Q2, K2, V2 = two(r(q), r(q)), two(r(k), r(_cx_J(k))), two(r(v), r(_cx_J(v)))
    g2 = two(g.unsqueeze(-1), torch.ones_like(g).unsqueeze(-1)).squeeze(-1)
    bw2 = two(bw.unsqueeze(-1), bw.unsqueeze(-1)).squeeze(-1)
    be2 = two(be.unsqueeze(-1), be.unsqueeze(-1)).squeeze(-1)
    read, S = pam_delta_batched(Q2, K2, V2, g2, bw2, be2, carry)
    return read.reshape(N, T, 2, 2 * K)[:, :, 1].reshape(N, T, K, 2), S


def cx_delta_step(q, k, v, g, bw, be, carry=None):
    """The true-complex reference: S <- g S (I - b_e k k^H) + b_w v k^H, y = S q, fp32; carry / returned S realified."""
    N, T, K, _ = q.shape
    c = lambda x: torch.view_as_complex(x.float().contiguous())
    qc, kc, vc = c(q), c(k), c(v)
    S = _complexify(carry) if carry is not None else torch.zeros(N, K, K, dtype=torch.complex64, device=q.device)
    reads = []
    for t in range(T):
        kt, vt, qt = kc[:, t], vc[:, t], qc[:, t]
        Sk = torch.einsum('nvk,nk->nv', S, kt)
        S = g[:, t, None, None].float() * (S - be[:, t, None, None].float() * Sk.unsqueeze(-1) * kt.conj().unsqueeze(-2)) \
            + bw[:, t, None, None].float() * vt.unsqueeze(-1) * kt.conj().unsqueeze(-2)
        reads.append(torch.einsum('nvk,nk->nv', S, qt))
    return torch.view_as_real(torch.stack(reads, dim=1)), _realify(S)


class CxGatedUnit(ComplexGatedUnit):
    """ComplexGatedUnit with an explicit hidden width (M1 param matching); hidden = 0 keeps dim * expand."""

    def __init__(self, feature: Dim, expand: int = 3, activation: str = 'modrelu', pair: Dim | None = None, hidden: int = 0):
        nn.Module.__init__(self)
        self.feature = feature
        self.hidden = Dim("hidden", hidden or feature.size * expand)
        self.complex_pair = pair if pair is not None else Dim("complex_pair", 2)
        self.policy = SplitComplex(self.complex_pair)
        self.gate_proj = ComplexLinear(self.feature, self.hidden, bias=False, pair=self.complex_pair)
        self.up_proj = ComplexLinear(self.feature, self.hidden, bias=False, pair=self.complex_pair)
        self.down_proj = ComplexLinear(self.hidden, self.feature, bias=False, pair=self.complex_pair)
        self.act = _build_activation(activation, self.hidden, self.complex_pair)


class CxPAMLayer(PAMLayer):
    """The v13 complex PAMLayer + delta / chrono / out_gate / reset / M2 exposure (see the module doc)."""

    def __init__(self, cfg: PAMConfig, layer_idx: int = 0, model_dim: Dim | None = None, complex_pair: Dim | None = None):
        super().__init__(cfg, layer_idx=layer_idx, model_dim=model_dim, complex_pair=complex_pair)
        # Complex ports of the real arm's parts (cx_m2, user via Lead 2026-11-05), each identity / off at init like its twin.
        # The gates read the token's real+imag concat (like dt_proj). A3 delta: unit complex keys,
        # S_t = g_t S (I - b_e k k^H) + b_w v k^H, y = S q; prefill runs the REAL fused delta kernel on the realified notebook
        # (two orthonormal rank-1 sub-steps per token: keys r(k), J r(k)), decode is a true complex stepwise reference.
        self.delta = cfg.delta
        if self.delta:
            self.erase_proj = NamedLinear(self.real_imag_feature, self.decay_out)
            self.write_proj = NamedLinear(self.real_imag_feature, self.decay_out)
        # N1 Chrono: per-head warp g = exp(clamp(W x, +/-3)) on the complex RoPE angle; exclusive cumulative clock; zero-init W
        # => exactly the complex RoPE (same inv_freq as build_rope_cache).
        self.chrono = cfg.chrono
        if self.chrono:
            self.register_buffer(
                'chrono_inv_freq',
                1.0 / (10000.0 ** (torch.arange(cfg.head_dim).float() / cfg.head_dim)),
                persistent=False,
            )
            self.warp_proj = NamedLinear(self.real_imag_feature, self.decay_out)
            self.warp_proj._zero_init = True
        # N4 read-out gate silu(W x + b), zero-init W, b = silu^-1(1): exactly 1 at start.
        self.out_gate = cfg.out_gate
        if self.out_gate:
            self.gate_proj = NamedLinear(self.real_imag_feature, self.decay_out)
            self.gate_proj._zero_init = True
            self.gate_proj._init_bias = 1.2785
        # M2: a notebook other blocks read (cfg.share_reads) exposes its realified delta write + its Chrono rotation.
        self.expose = cfg.share_reads
        self._shared = None
        self._rot = None


    def _real_imag(self, tokens: NamedTensor) -> NamedTensor:
        return to_real_concat(tokens, into=self.real_imag_feature)


    def _head_gate(self, proj, tokens: NamedTensor) -> torch.Tensor:
        """A per-token, per-head scalar from the token's real+imag, raw [B, H, T]."""
        batch, time = tokens.layout[0], tokens.layout[1]
        return proj(self._real_imag(tokens)).alias(self.decay_out, self.heads).raw(batch, self.heads, time)


    def _project(self, tokens: NamedTensor, step_offset: int, clock=None):
        """Query, key, value on (batch, heads, time, head_feature, complex_pair), + the Chrono clock out.

        The fused QKV axis splits with its factors at the end of the layout —
        the one order sempty's named split accepts — then heads move before
        time for the notebook. RoPE rotates query and key by their position,
        so the notebook's associations carry word order. With ``chrono`` the
        rotation angle is the learned clock (identical to RoPE at init).
        """
        batch, time = tokens.layout[0], tokens.layout[1]
        qkv = self.qkv_proj(tokens)
        qkv = qkv.to(batch, time, self.qkv_slot, self.heads, self.head_feature,
                     self.complex_pair)
        queries, keys, values = (qkv.select(self.qkv_slot, slot) for slot in (0, 1, 2))

        if self.use_rope and self.chrono:
            # Raw-torch boundary (cumsum over time), mirrors RealPAMLayer._rotate_learned in complex form: e^{i phi} q.
            B, H, T, K = batch.size, self.heads.size, time.size, self.head_dim
            g = torch.exp(self._head_gate(self.warp_proj, tokens).float().clamp(-3.0, 3.0))   # [B,H,T]
            run = torch.cumsum(g, dim=2)
            if clock is not None:
                run = run + clock.to(run.dtype).unsqueeze(-1)
            tau = run - g                                                     # exclusive: tau_0 = clock
            clock_out = run[:, :, -1]
            phi = tau.unsqueeze(-1) * self.chrono_inv_freq.to(tau.device).view(1, 1, 1, K)   # [B,H,T,K]
            cos_h, sin_h = torch.cos(phi).to(tokens.dtype), torch.sin(phi).to(tokens.dtype)   # [B,H,T,K]
            if self.expose:
                self._rot = (cos_h, sin_h)                                     # M2: readers use the writer's clock
            cos, sin = cos_h.permute(0, 2, 1, 3), sin_h.permute(0, 2, 1, 3)   # [B,T,H,K]
            def _rot(x):
                r = x.raw(batch, time, self.heads, self.head_feature, self.complex_pair)
                a, b = r[..., 0], r[..., 1]
                out = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
                return named(out, (batch, time, self.heads, self.head_feature, self.complex_pair), self.policy)  # named-exit
            layout = (batch, self.heads, time, self.head_feature, self.complex_pair)
            return (_rot(queries).to(*layout), _rot(keys).to(*layout), values.to(*layout), clock_out)

        if self.use_rope:
            position_end = step_offset + tokens.size(time)
            if position_end > self.rope_cache.shape[0]:
                self.register_buffer(
                    'rope_cache',
                    build_rope_cache(position_end * 2, self.head_dim).to(tokens.device),
                    persistent=False,
                )
            rotation = named(
                self.rope_cache[step_offset:position_end].to(dtype=tokens.dtype),
                (time, self.head_feature, self.complex_pair), self.policy,
            )  # named-exit: the RoPE table is a plain buffer, sliced here
            queries = queries * rotation
            keys = keys * rotation

        layout = (batch, self.heads, time, self.head_feature, self.complex_pair)
        return (queries.to(*layout), keys.to(*layout), values.to(*layout), None)


    # ── A3 delta, complex (raw-torch boundary) ───────────────────────────────
    # Realified notebook R(S) [2K, 2K] on interleaved (re, im) vectors r(z): for a unit complex key k, the complex projector k k^H
    # realifies to r(k) r(k)^T + Jr(k) Jr(k)^T (J = multiplication by i), two ORTHONORMAL rank-1 projectors, so one complex
    # erase/write step equals two real delta sub-steps (decay g on the first, 1 on the second; keys r(k), Jr(k); values r(v),
    # Jr(v)); the read y = R(S) r(q) of the second sub-step is the complex read. Exact; parity: v13_sempty/tmp/cx_delta_parity.py.

    @staticmethod
    def _cx_J(x):                    # [..., K, 2] -> i * x, same layout
        return torch.stack((-x[..., 1], x[..., 0]), dim=-1)

    @staticmethod
    def _cx_unit(k):                 # unit complex vectors over (K, pair)
        n = k.float().pow(2).sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12).sqrt()
        return (k.float() / n).to(k.dtype)

    @staticmethod
    def _realify(S):                 # complex [..., K, K] -> realified [..., 2K, 2K] (interleaved)
        *lead, K, _ = S.shape
        R = torch.zeros(*lead, 2 * K, 2 * K, dtype=S.real.dtype, device=S.device)
        R[..., 0::2, 0::2] = S.real; R[..., 0::2, 1::2] = -S.imag
        R[..., 1::2, 0::2] = S.imag; R[..., 1::2, 1::2] = S.real
        return R

    @staticmethod
    def _complexify(R):              # realified [..., 2K, 2K] -> complex [..., K, K]
        return torch.complex(R[..., 0::2, 0::2].float(), R[..., 1::2, 0::2].float())

    def _cx_inputs(self, tokens, queries, keys, values, decay):
        batch, time = tokens.layout[0], tokens.layout[1]
        lay = (batch, self.heads, time, self.head_feature, self.complex_pair)
        q, k, v = queries.raw(*lay), self._cx_unit(keys.raw(*lay)), values.raw(*lay)     # [B,H,T,K,2]
        g = decay.raw(batch, self.heads, time)
        keep = getattr(self, '_keep', None)          # RESET_AT_EOS: retention 0 at document separators
        if keep is not None:
            B, H, T = g.shape
            g = g * keep[:, None, :].expand(B, H, T).to(g.dtype)
        be = torch.sigmoid(self._head_gate(self.erase_proj, tokens) - 3.0).clamp(max=0.95)
        bw = torch.sigmoid(self._head_gate(self.write_proj, tokens))
        return q, k, v, g, bw, be

    def _chunked_delta(self, tokens, queries, keys, values, decay, state=None):
        batch, time = tokens.layout[0], tokens.layout[1]
        B, H, T, K = batch.size, self.heads.size, time.size, self.head_dim
        q, k, v, g, bw, be = self._cx_inputs(tokens, queries, keys, values, decay)
        r = lambda x: x.reshape(B, H, T, 2 * K)
        two = lambda a, b: torch.stack((a, b), dim=3).reshape(B * H, 2 * T, *a.shape[3:])
        Q2 = two(r(q), r(q)); K2 = two(r(k), r(self._cx_J(k))); V2 = two(r(v), r(self._cx_J(v)))
        g2 = two(g.unsqueeze(-1), torch.ones_like(g).unsqueeze(-1)).squeeze(-1)
        bw2 = two(bw.unsqueeze(-1), bw.unsqueeze(-1)).squeeze(-1)
        be2 = two(be.unsqueeze(-1), be.unsqueeze(-1)).squeeze(-1)
        carry = state.reshape(B * H, 2 * K, 2 * K).float() if state is not None else None
        if self.expose:
            read, S, write = pam_delta_batched(Q2, K2, V2, g2, bw2, be2, carry, return_write=True)
            self._shared = {'write': write, 'rot': self._rot}
        else:
            read, S = pam_delta_batched(Q2, K2, V2, g2, bw2, be2, carry)
        read = read.reshape(B, H, T, 2, 2 * K)[:, :, :, 1].reshape(B, H, T, K, 2)
        output = named(read * (self.head_dim ** -0.5),
                       (batch, self.heads, time, self.head_row, self.complex_pair), self.policy)  # named-exit
        return output, S.reshape(B, H, 2 * K, 2 * K)

    def _stepwise_delta(self, tokens, queries, keys, values, decay, state):
        """True complex reference: S <- g S (I - b_e k k^H) + b_w v k^H, y = S q (fp32), state carried realified."""
        batch, time = tokens.layout[0], tokens.layout[1]
        B, H, T, K = batch.size, self.heads.size, time.size, self.head_dim
        q, k, v, g, bw, be = self._cx_inputs(tokens, queries, keys, values, decay)
        c = lambda x: torch.view_as_complex(x.float().contiguous())                     # [B,H,T,K]
        qc, kc, vc = c(q), c(k), c(v)
        S = (self._complexify(state.reshape(B, H, 2 * K, 2 * K)) if state is not None
             else torch.zeros(B, H, K, K, dtype=torch.complex64, device=q.device))
        reads = []
        for t in range(T):
            kt, vt, qt = kc[:, :, t], vc[:, :, t], qc[:, :, t]
            Sk = torch.einsum('bhvk,bhk->bhv', S, kt)
            S = g[:, :, t, None, None].float() * (S - be[:, :, t, None, None].float() * Sk.unsqueeze(-1) * kt.conj().unsqueeze(-2)) \
                + bw[:, :, t, None, None].float() * vt.unsqueeze(-1) * kt.conj().unsqueeze(-2)
            reads.append(torch.einsum('bhvk,bhk->bhv', S, qt))
        y = torch.view_as_real(torch.stack(reads, dim=2)).to(q.dtype) * (self.head_dim ** -0.5)   # [B,H,T,K,2]
        if self.expose:                                  # M2 decode: readers read S_t (complex)
            if T != 1:
                raise NotImplementedError("M2 stepwise writer exposes one step at a time")
            self._shared = {'state': S, 'rot': self._rot}
        output = named(y, (batch, self.heads, time, self.head_row, self.complex_pair), self.policy)  # named-exit
        return output, self._realify(S)

    def forward(self, x: NamedTensor, state=None, step_offset: int = 0):
        tokens = self._as_token(x)
        seq_len = tokens.size(tokens.layout[1])
        if self.chrono and state is not None:          # Chrono carries (notebook, clock)
            state, clock = state
        else:
            clock = None
        queries, keys, values, clock_out = self._project(tokens, step_offset, clock)
        decay = self._decay(tokens)

        if self.delta and seq_len > 1:
            output, new_state = self._chunked_delta(tokens, queries, keys, values, decay, state)
        elif self.delta:
            output, new_state = self._stepwise_delta(tokens, queries, keys, values, decay, state)
        elif state is None and seq_len > 1:
            output, new_state = self._chunked(tokens, queries, keys, values, decay)
        else:
            output, new_state = self._stepwise(tokens, queries, keys, values, decay, state)

        if self.out_gate:                              # N4: per-token, per-head real gate on the complex read
            batch, time = tokens.layout[0], tokens.layout[1]
            gate = torch.nn.functional.silu(self._head_gate(self.gate_proj, tokens))      # [B,H,T]
            out_raw = output.raw(batch, self.heads, time, self.head_row, self.complex_pair)
            output = named(out_raw * gate[..., None, None].to(out_raw.dtype),
                           (batch, self.heads, time, self.head_row, self.complex_pair), self.policy)  # named-exit

        out = self.o_proj(output.alias(self.head_row, self.head_feature)
                          .to(tokens.layout[0], tokens.layout[1], self.inner,
                              self.complex_pair))
        if self.training:
            out = out * as_complex_dropout_mask(self.out_dropout, out)
        carried = new_state if isinstance(new_state, torch.Tensor) or new_state is None else new_state.data  # named-exit
        if self.chrono:
            return out, (carried, clock_out)
        return out, carried


class CxBlock(nn.Module):
    """Pre-norm residual: gated channel mix, then PAM sequence mix."""

    def __init__(self, cfg: PAMConfig, layer_idx: int = 0,
                 model_dim: Dim | None = None, complex_pair: Dim | None = None):
        super().__init__()
        self.model_dim = model_dim or Dim("model_dim", cfg.dim)
        self.complex_pair = complex_pair or Dim("complex_pair", 2)
        self.dropout = cfg.dropout
        self.norm1 = ComplexNorm(self.model_dim, pair=self.complex_pair)
        self.cgu = CxGatedUnit(self.model_dim, cfg.expand,
                                    activation=cfg.activation,
                                    pair=self.complex_pair, hidden=cfg.mlp_hidden)
        self.cgu_dropout = nn.Dropout(cfg.dropout)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        # M1 (complex port, cx_m2): a block outside cfg.pam_layers is a pure gated-MLP block.
        self.has_pam = not cfg.pam_layers or layer_idx in tuple(cfg.pam_layers)
        self.pam = None
        if self.has_pam:
            self.norm2 = ComplexNorm(self.model_dim, pair=self.complex_pair)
            self.pam = CxPAMLayer(cfg, layer_idx=layer_idx,
                                model_dim=self.model_dim, complex_pair=self.complex_pair)
            # PAM starts soft: the memory path learns while the residual carries.
            self.pam_scale = nn.Parameter(torch.tensor(0.1))
        # M2: a notebook-less block above the first notebook reads the nearest notebook below it.
        self.reader = None
        if (not self.has_pam and cfg.share_reads
                and layer_idx > min(tuple(cfg.pam_layers))):
            self.norm2 = ComplexNorm(self.model_dim, pair=self.complex_pair)
            self.reader = CxSharedRead(cfg, self.model_dim, self.complex_pair)
        self.window = None

    def forward(self, x: NamedTensor, pam_state=None, step_offset: int = 0, shared=None):
        cgu_out = self.cgu(self.norm1(x))
        if self.training:
            cgu_out = cgu_out * as_complex_dropout_mask(self.cgu_dropout, cgu_out)
        x = x + cgu_out * self.cgu_scale
        if not self.has_pam:
            if self.reader is not None:
                x = x + self.reader(self.norm2(x), shared)
            return x, None
        pam_out, new_state = self.pam(self.norm2(x), state=pam_state,
                                      step_offset=step_offset)
        return x + pam_out * self.pam_scale, new_state


class CxSharedRead(nn.Module):
    """M2, complex: a cheap reader of another block's (realified) delta notebook.

        y_t = gate_t * (S^w_t . rot_w(q_t)) / sqrt(d),   out = W_o y   (complex W_o = 0 at init)

    The query is complex, rotated by the writer's Chrono clock e^{i phi_w}; the read runs on the writer's realified WY write with
    the query on both sub-steps (the second sub-step's read is the complex read); decode reads the writer's complex S_t.
    """

    def __init__(self, cfg: PAMConfig, model_dim: Dim, complex_pair: Dim):
        super().__init__()
        self.model_dim, self.complex_pair = model_dim, complex_pair
        self.policy = SplitComplex(complex_pair)
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.heads = Dim("heads", cfg.n_heads)
        self.inner = Dim("reader_inner", cfg.n_heads * cfg.head_dim)
        self.real_imag_feature = Dim("real_imag_feature", cfg.dim * 2)
        self.q_proj = ComplexLinear(model_dim, self.inner, bias=False, pair=complex_pair)
        self.o_proj = ComplexLinear(self.inner, model_dim, bias=False, pair=complex_pair)
        with torch.no_grad():                               # identity at start (ComplexLinear is untouched by _init_weights)
            self.o_proj.weight_real.zero_(); self.o_proj.weight_imag.zero_()
        self.gate_out = Dim("decay_out", cfg.n_heads)
        self.gate = NamedLinear(self.real_imag_feature, self.gate_out)
        self.gate._zero_init = True
        self.gate._init_bias = 1.2785                       # silu(1.2785) = 1

    def forward(self, x: NamedTensor, shared: dict) -> NamedTensor:
        batch, time = x.layout[0], x.layout[1]
        B, T, H, K = batch.size, time.size, self.n_heads, self.head_dim
        q = self.q_proj(x).raw(batch, time, self.inner, self.complex_pair).reshape(B, T, H, K, 2).transpose(1, 2)  # [B,H,T,K,2]
        cos, sin = shared['rot']                                                   # [B,H,T,K]
        a, b = q[..., 0], q[..., 1]
        q = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
        if 'state' in shared:                                                      # decode: complex S_t q_t
            S = shared['state']                                                    # [B,H,K,K] complex
            read = torch.view_as_real(torch.einsum('bhvk,bhtk->bhtv', S, torch.view_as_complex(q.float().contiguous())))
        else:
            rq = q.reshape(B, H, T, 2 * K)
            Q2 = torch.stack((rq, rq), dim=3).reshape(B * H, 2 * T, 2 * K)
            read = pam_shared_read(Q2, shared['write']).reshape(B, H, T, 2, 2 * K)[:, :, :, 1].reshape(B, H, T, K, 2)
        read = read.to(q.dtype) * (K ** -0.5)
        g = torch.nn.functional.silu(
            self.gate(to_real_concat(x, into=self.real_imag_feature)).alias(self.gate_out, self.heads).raw(batch, self.heads, time))
        y = (read * g[..., None, None].to(read.dtype)).transpose(1, 2).reshape(B, T, H * K, 2)
        y = named(y, (batch, time, self.inner, self.complex_pair), self.policy)   # named-exit: shared-read boundary
        return self.o_proj(y)


class CxLM(LM):
    """The v13 LM with CxBlocks in the complex arm (the base LM asserts M1 / M2 are real-only, so it is built on a copy of the
    config without them, then the blocks are replaced and re-initialised)."""

    def __init__(self, cfg: PAMConfig):
        assert cfg.is_complex, "CxLM is the complex arm"
        base = copy.copy(cfg)
        base.pam_layers, base.share_reads, base.mlp_hidden, base.echo = (), False, 0, False
        super().__init__(base)
        self.config = cfg
        # E1 echo, complex: q, k = complex W h; gates from h's real+imag; values = the token's complex embedding (RMS-unit).
        self.echo = cfg.echo
        if self.echo:
            assert cfg.dim == cfg.n_heads * cfg.head_dim, "echo heads must tile dim"
            self.echo_qk_out = Dim("echo_qk_out", 2 * cfg.dim)
            self.echo_qk = ComplexLinear(self.model_dim, self.echo_qk_out, bias=False, pair=self.complex_pair)
            self.echo_gate_out = Dim("echo_gate_out", 3 * cfg.n_heads)        # dt | write | erase
            self.echo_gates = NamedLinear(self.real_imag_feature, self.echo_gate_out)
            self.echo_inner = Dim("echo_inner", cfg.dim)
            self.echo_out = ComplexLinear(self.echo_inner, self.model_dim, bias=False, pair=self.complex_pair)
        self.blocks = nn.ModuleList([CxBlock(cfg, layer_idx=i, model_dim=self.model_dim, complex_pair=self.complex_pair)
                                     for i in range(cfg.n_layers)])
        self._init_weights()
        with torch.no_grad():
            for b in self.blocks:                 # CxSharedRead's complex out-proj: zero (identity) again after _init_weights
                if b.reader is not None:
                    b.reader.o_proj.weight_real.zero_(); b.reader.o_proj.weight_imag.zero_()
            if self.echo:                         # echo starts as the identity (zero complex out-proj), retention ~ 0.9975
                self.echo_out.weight_real.zero_(); self.echo_out.weight_imag.zero_()
                self.echo_gates.bias[:cfg.n_heads].fill_(cfg.echo_dt_bias)

    def _echo_read(self, hidden: NamedTensor, input_ids, batch: Dim, time: Dim, state=None):
        """E1 echo, complex (raw-torch boundary), the real arm's _echo_read with complex q / k / v:

            S_t = g_t S_{t-1} (I - b_e k_t k_t^H) + b_w v_t k_t^H,  r_t = S_t q_t
            k_t = unit(W_k h_{t-1}),  v_t = rms_unit(E_r[x_t] + i E_i[x_t]),  q_t = W_q h_t

        Prefill: the realified fused delta kernel (cx_delta_chunked); decode: the complex reference (cx_delta_step). ``state`` =
        (S realified [B, H, 2K, 2K] fp32, h_prev [B, D, 2]). Per-head complex RMS-norm, then the zero-init complex echo_out.
        """
        cfg = self.config
        h = hidden.raw(batch, time, self.model_dim, self.complex_pair)            # [B,T,D,2]
        B, T, D, _ = h.shape
        H, K = cfg.n_heads, cfg.head_dim
        qk = self.echo_qk(hidden).raw(batch, time, self.echo_qk_out, self.complex_pair)
        q, k = qk[:, :, :D], qk[:, :, D:]                                           # [B,T,D,2]
        if state is None:
            k = nn.functional.pad(k, (0, 0, 0, 0, 1, 0))[:, :T]                     # k_t from h_{t-1}; k_0 = 0
        else:
            hp = named(state[1].to(h.dtype)[:, None], (batch, Dim("time1", 1), self.model_dim, self.complex_pair), self.policy)
            k_prev = self.echo_qk(hp).raw(batch, hp.layout[1], self.echo_qk_out, self.complex_pair)[:, 0, D:]
            k = torch.cat([k_prev[:, None], k[:, :T - 1]], dim=1)
        v = torch.stack((self.embed.embed_real(input_ids), self.embed.embed_imag(input_ids)), dim=-1).to(h.dtype)   # [B,T,D,2]
        v = v * torch.rsqrt(v.float().pow(2).sum(-1).mean(-1, keepdim=True) + 1e-6)[..., None].to(h.dtype)
        heads = lambda x: x.reshape(B, T, H, K, 2).transpose(1, 2).reshape(B * H, T, K, 2)
        per_head = lambda x: x.transpose(1, 2).reshape(B * H, T)
        gates = self.echo_gates(to_real_concat(hidden, into=self.real_imag_feature)).raw(batch, time, self.echo_gate_out)
        dt, wr, er = gates.split(H, dim=-1)
        retention = per_head(torch.exp(-nn.functional.softplus(dt)))
        bw = per_head(torch.sigmoid(wr))
        be = per_head(torch.sigmoid(er - 3.0).clamp(max=0.95))
        kn = _cx_unit(heads(k))
        dil = getattr(self, '_dilate', None)                                         # clk eval hook on the echo memory too
        if dil is not None and (len(dil) < 4 or dil[3] in ('all', 'echo')):
            d0, d1, sc = dil[:3]
            retention = retention.clone()
            retention[:, d0:d1] = retention[:, d0:d1].clamp_min(1e-12) ** (1.0 / sc)
        if getattr(cfg, 'reset_at_eos', False):
            sep = (input_ids == cfg.reset_token_id)
            retention = retention * (~sep).to(retention.dtype)[:, None, :].expand(B, H, T).reshape(B * H, T)
            kn = kn * (~sep).to(kn.dtype)[:, None, :, None, None].expand(B, H, T, 1, 1).reshape(B * H, T, 1, 1)
        carry = state[0].reshape(B * H, 2 * K, 2 * K).float() if state is not None else None
        if state is None:
            read, S = cx_delta_chunked(heads(q), kn, heads(v), retention, bw, be, None)
        else:
            read, S = cx_delta_step(heads(q), kn, heads(v), retention, bw, be, carry)
        read = read.float()
        read = read * torch.rsqrt(read.pow(2).sum(-1).mean(-1, keepdim=True) + 1e-6)[..., None]   # complex RMS over K
        read = read.to(h.dtype).reshape(B, H, T, K, 2).transpose(1, 2).reshape(B, T, D, 2)
        out = self.echo_out(named(read, (batch, time, self.echo_inner, self.complex_pair), self.policy))  # named-exit: echo
        return out, (S.reshape(B, H, 2 * K, 2 * K), h[:, -1].detach().float())
