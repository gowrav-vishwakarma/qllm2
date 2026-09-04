"""
v13_sempty — a simple, pure Phase-Associative Memory (PAM) language model
on the sempyt named-axis framework.

What this model is
------------------

Every layer carries one fixed-size *notebook* per head that remembers
everything seen so far, one association at a time:

    notebook_t  =  decay_t * notebook_{t-1}  +  value_t (x) key_t
    read_t      =  scale * ( notebook_t . query_t )

In the complex model the notebook is a d×d complex matrix and the write
conjugates the key (so a later read with the same key recalls the value).
The fully-real model (``is_complex = False``) keeps one real width and runs
the identical recurrence in plain real arithmetic — a single GEMM, no phase,
no conjugates — with the same RoPE word-order carrier.

In words, each token:

  1. fades what the notebook holds, by a learned per-head number decay_t in
     (0, 1);
  2. writes one new association — the token's value, hung on the phase of its
     key (the write is conjugate, so it can later be recalled exactly);
  3. reads what the notebook now says about its query.

That is the whole memory. No gates, no erasures, no extra states: the decay
is the only learned knob on the notebook, and it is a single number per head.

Position comes from RoPE, applied to the query and the key. The recurrence
alone only knows recency (a decaying number); word order needs a position
carrier, and RoPE is the free one — a buffer and one complex multiply, no
parameters. Every lean variant in v7/v11 kept it; learned input embeddings
were measured neutral there, so they stay out (see EXPERIMENTS_SEMPY.md).

Training and decoding run the same algebra two ways:

  * train / prefill  — time is processed in windows of ``chunk_size``. Inside
    a window the notebook is built from the bounded retention matrix
    ``M[s, t] = a_s / a_t <= 1`` (``a_s`` the product of the decays up to s,
    ``M`` computed in log space as ``exp(C_t - C_s)`` so the backward stays
    bounded — the naive factored form ``a_s * cumsum(write / a_s)`` overflows
    it once a window's retention decays toward 0):

        notebook_s = sum_{t <= s} M[s, t] * write_t   (+ a_s * notebook_in)

    The notebook at the window's end is carried into the next.
  * decode           — the same recurrence, one step per token, on the
    carried notebook. O(1) per token, independent of context length.

The two agree to round-off (selftest ``test_parallel_vs_recurrent``).

Style
-----

The code speaks in the model's own words: ``decay``, ``write``, ``notebook``,
``read``, ``rotation``. All layout is named — ``.to()`` / ``.alias()`` /
``contract`` / ``outer`` — and ``check_torch_layout.py`` fails the build if
anything reaches for ``view`` / ``permute`` / ``[..., 0]``. Raw torch appears
only at the declared boundaries: the RoPE table (built once, outside the
graph), the stable notebook scan, the tied-logit hand-off, the chunked-CE
Function, and the sampling loop.

Named axes
----------
  batch             sequence items in the minibatch
  time              token positions
  model_dim         residual / embedding width
  heads             PAM heads
  head_feature      per-head channel width (d)
  head_pair         real RoPE rotation pairs (head_feature // 2)
  complex_pair      last axis of size 2: real then imag
  qkv_slot / qkv_fused   fused Q / K / V packing
  head_row / head_col    the two axes of the d x d notebook
  chunk_time        token positions inside one window
  real_imag_feature  concat(real, imag) along model_dim
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from sempyt.dim import Dim, ProductDim
from sempyt.nn import Linear as NamedLinear
from sempyt.nn import RMSNorm
from sempyt.ops import at, contract, imag, outer, real
from sempyt.policies import SplitComplex
from sempyt.structural import (
    cat, exp, flatten, select, silu, softplus, stack, take, zeros,
)
from sempyt.tensor import NamedTensor, named

from v13_sempty.config import PAMConfig
from v13_sempty.complex_ops import (
    ComplexEmbed,
    ComplexGatedUnit,
    ComplexLinear,
    ComplexNorm,
    as_complex_dropout_mask,
    build_rope_cache,
    to_real_concat,
)
from v13_sempty.real_ops import (
    RealEmbed,
    RealGatedUnit,
    RealNorm,
    build_rope_cache_real,
)
from v13_sempty.triton_kernels import (
    fused_real_pam_read,
    fused_complex_pam_read,
    pam_delta_torch,
    kernel_enabled,
)

# Diagnostic hook (off by default): when a trainer sets this to a fresh list
# and flips PAMLayer.capture_decay, each _decay forward appends the realized
# per-head retention (detached, fp32, mean over batch+time) for its layer so
# the trainer can report which layer is retaining how much memory.
_retention_capture: list = []

def _capture_retention(retention: NamedTensor) -> None:
    """Append the realized per-head retention mean (over batch+time) to the
    module-level capture list. Caller must have set capture_decay=True."""
    r = retention.raw().float().mean(dim=(0, 2)).detach()  # named-exit: diagnostic read, detached, off-by-default
    _retention_capture.append(r)


def _stable_notebook(write_nt, decay_nt, carried_nt, window, head_row, head_col,
                     complex_pair, policy, dtype):
    r"""Chunk notebook in the stable log-space decay-matrix form.

    The notebook recurrence ``S_s = g_s S_{s-1} + write_s`` has the closed
    form ``S_s = a_s (S_in + sum_{j<=s} write_j / a_j)`` with
    ``a_s = prod_{j<=s} g_j``.  That factored form overflows the *backward*:
    the gradient of ``write_j / a_j`` is ``write_j / a_j**2``, which blows
    past fp32 (~1e40) once the learned retention decays a 256-window toward
    0 -- even though the forward notebook stays O(1).  Re-expressed with the
    decay matrix ``M[s, j] = a_s / a_j = exp(C_j - C_s) <= 1`` (``C =
    cumsum(-log g)``), every intermediate is bounded by the write magnitudes
    and no ``1 / retention`` term appears in the forward *or* backward.

    ``write_nt``    [B, H, window, row, col(, pair)]
    ``decay_nt``    [B, H, window]  in (0, 1)
    ``carried_nt``  [B, H, row, col(, pair)] or None
    returns a NamedTensor notebook [B, H, window, row, col(, pair)] in
    ``dtype``.  All scan math runs in fp32; the result is cast to ``dtype``.
    """
    write_raw = write_nt.raw().float()              # [B,H,T,row,col(,pair)] fp32
    decay_raw = decay_nt.raw().float()              # [B,H,T] fp32, in (0,1)
    carried_raw = carried_nt.raw().float() if carried_nt is not None else None
    B, H, T = write_raw.shape[:3]
    # C_s = sum_{t<=s} -log decay_t  (>= 0, non-decreasing in s).
    C = torch.cumsum(-torch.log(decay_raw + 1e-6), dim=-1)        # [B,H,T]
    # Retention M[s, t] = a_s / a_t = exp(C_t - C_s) (t <= s), bounded by 1.
    # The FULL [s, t] exponent is computed before the tril mask, and the
    # anti-causal part (t > s) is POSITIVE, growing to C_end.  As training
    # drives retention down, C_end crosses log(fp32_max) ~= 88.7, exp
    # overflows there, and inf * 0 (the tril mask) = NaN poisons every
    # notebook row (the einsum sums over all t).  Clamping the exponent at
    # 0 keeps exp <= 1, is a no-op on the causal triangle (t <= s, E <= 0),
    # and gives the anti-causal region exactly-zero value AND gradient.
    M = torch.exp(torch.clamp(C.unsqueeze(-2) - C.unsqueeze(-1), max=0.0))
    M = M * torch.tril(torch.ones(T, T, device=write_raw.device))
    # notebook_s = sum_{t<=s} M[s, t] write_t  (weighted sum over t; a plain
    # cumsum would only be valid for an unweighted causal mask).
    rest = write_raw.shape[3:]
    acc = torch.einsum('bhst,bhtk->bhsk', M, write_raw.reshape(B, H, T, -1))
    if carried_raw is not None:
        # The carried-in notebook also decays inside the window:
        # a_s * S_in with a_s = prod_{t<=s} decay_t = exp(-C_s).
        acc = acc + torch.exp(-C).unsqueeze(-1) * carried_raw.reshape(B, H, -1).unsqueeze(2)
    layout = (write_nt.layout[0], write_nt.layout[1], window, head_row, head_col) \
        + ((complex_pair,) if complex_pair is not None else ())
    return named(acc.reshape(B, H, T, *rest).to(dtype), layout, policy)
# ── Phase-Associative Memory ─────────────────────────────────────────────────


class PAMLayer(nn.Module):
    r"""One head-fan of phase-associative memories.

    Per head, a complex d×d notebook ``S`` carries the whole past:

        S_t = decay_t * S_{t-1} + value_t (x) conj(key_t)
        y_t = d^{-1/2} * (S_t . query_t)

    ``decay_t = exp(-softplus(linear(real+imag of token) + bias))`` is the
    only learned memory quantity — a single number per head per token. The
    write conjugates the key (so a later read with the same key recalls the
    value); the read uses the raw query. Reads are *after* the write, so a
    token can recall what it just wrote.

    ``forward`` returns the PAM output ``[batch, time, inner]`` (inner =
    heads × head_feature, folded back to model_dim by the output projection)
    and the notebook to carry on (raw ``[batch, heads, d, d, 2]`` per
    ``state``; ``None`` when nothing was carried in and nothing should be
    returned).
    """

    def __init__(self, cfg: PAMConfig, layer_idx: int = 0,
                 model_dim: Dim | None = None, complex_pair: Dim | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.model_dim = model_dim or Dim("model_dim", cfg.dim)
        self.complex_pair = complex_pair or Dim("complex_pair", 2)
        self.policy = SplitComplex(self.complex_pair)
        self.heads = Dim("heads", cfg.n_heads)
        self.head_feature = Dim("head_feature", cfg.head_dim)
        self.head_row = Dim("head_row", cfg.head_dim)
        self.head_col = Dim("head_col", cfg.head_dim)
        self.inner = ProductDim(self.heads, self.head_feature)
        self.real_imag_feature = Dim("real_imag_feature", cfg.dim * 2)

        # Fused QKV: one projection, then split. The three share the same
        # width, so one matrix is the whole fan.
        self.qkv_slot = Dim("qkv_slot", 3)
        self.qkv_fused = Dim("qkv_fused", 3 * cfg.n_heads * cfg.head_dim)
        self.qkv_proj = ComplexLinear(self.model_dim, self.qkv_fused,
                                      bias=False, pair=self.complex_pair)
        self.o_proj = ComplexLinear(self.inner, self.model_dim,
                                    bias=False, pair=self.complex_pair)

        # The decay: one number per head, read from the token's real+imag.
        self.decay_out = Dim("decay_out", cfg.n_heads)
        self.dt_proj = NamedLinear(self.real_imag_feature, self.decay_out)
        self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads) + cfg.base_dt_bias)

        # RoPE table, built once outside the graph (declared boundary).
        if cfg.use_rope:
            self.register_buffer(
                'rope_cache',
                build_rope_cache(cfg.max_seq_len, cfg.head_dim),
                persistent=False,
            )
        self.use_rope = cfg.use_rope

        self.chunk_size = cfg.chunk_size
        self.head_dim = cfg.head_dim
        self.out_dropout = nn.Dropout(cfg.dropout)
        # Diagnostic hook: when True (and a module-level _retention_capture
        # list is set), stash the realized per-head retention of the last
        # forward so the trainer can report it. Off by default (zero cost).
        self.capture_decay = False

    # ── small named helpers ──────────────────────────────────────────────────

    def _as_token(self, x: NamedTensor) -> NamedTensor:
        """Re-state the token axes under this layer's own Dim identities.

        Each forward wraps its batch/time in fresh Dims, and a carried
        notebook keeps the Dims of the call that created it; restating is
        an identity change only — no data moves.
        """
        batch, time = x.layout[0], x.layout[1]
        return named(x.data, (batch, time, self.model_dim, self.complex_pair),  # named-exit: restate
                     self.policy)

    def _as_notebook(self, state, batch: Dim) -> NamedTensor:
        """Re-state a carried notebook onto this call's axes (no data moves)."""
        layout = (batch, self.heads, self.head_row, self.head_col, self.complex_pair)
        data = state.data if isinstance(state, NamedTensor) else state  # named-exit: restate
        return named(data, layout, self.policy)

    def _project(self, tokens: NamedTensor, step_offset: int):
        """Query, key, value on (batch, heads, time, head_feature, complex_pair).

        The fused QKV axis splits with its factors at the end of the layout —
        the one order sempty's named split accepts — then heads move before
        time for the notebook. RoPE rotates query and key by their position,
        so the notebook's associations carry word order.
        """
        batch, time = tokens.layout[0], tokens.layout[1]
        qkv = self.qkv_proj(tokens)
        qkv = qkv.to(batch, time, self.qkv_slot, self.heads, self.head_feature,
                     self.complex_pair)
        queries, keys, values = (qkv.select(self.qkv_slot, slot) for slot in (0, 1, 2))

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
        return (queries.to(*layout), keys.to(*layout), values.to(*layout))

    def _decay(self, tokens: NamedTensor) -> NamedTensor:
        """Per-head retention in (0, 1): exp(-softplus(linear(token) + bias))."""
        batch, time = tokens.layout[0], tokens.layout[1]
        real_imag = to_real_concat(tokens, into=self.real_imag_feature)
        logit = self.dt_proj(real_imag).alias(self.decay_out, self.heads)
        retention = exp(-softplus(logit + named(self.dt_bias, (self.heads,))))
        if self.capture_decay:
            _capture_retention(retention)
        return retention.to(batch, self.heads, time)

    # ── train / prefill: one closed form per window, notebook carried ────────

    def _chunked(self, tokens: NamedTensor, queries, keys, values,
                 decay) -> tuple[NamedTensor, NamedTensor]:
        # Fused path: the same closed form as _stable_notebook, run on the real
        # kernel via the split-real conjugate-score trick (never materialises
        # the [B,H,w,K,K,2] per-position notebook). Declared raw boundary.
        batch, time = tokens.layout[0], tokens.layout[1]
        if kernel_enabled() and tokens.data.is_cuda:
            return self._chunked_fused(batch, time, queries, keys, values, decay)

        seq_len = tokens.size(tokens.layout[1])
        carried = None  # the notebook coming in from earlier windows
        reads = []
        for start in range(0, seq_len, self.chunk_size):
            length = min(self.chunk_size, seq_len - start)
            window = Dim("chunk_time", length)

            write = outer(
                take(values, over=time, start=start, length=length, new=window)
                    .alias(self.head_feature, self.head_row),
                take(keys, over=time, start=start, length=length, new=window)
                    .alias(self.head_feature, self.head_col).conj(),
                (self.head_row, self.head_col),
            )
            window_decay = take(decay, over=time, start=start, length=length,
                                new=window)
            window_query = take(queries, over=time, start=start, length=length,
                                new=window).alias(self.head_feature, self.head_col)

            # Stable log-space decay-matrix notebook: the factored form
            # a_s * cumsum(write / a_s) overflows the backward (grad ~
            # write / a_s**2) once the learned retention decays a window
            # toward 0, so the notebook is built from the bounded decay
            # matrix instead (see _stable_notebook).
            window_notebook = _stable_notebook(
                write, window_decay, carried,
                window, self.head_row, self.head_col, self.complex_pair,
                self.policy, window_decay.dtype,
            )

            reads.append(
                contract(window_notebook, window_query, over=self.head_col)
                * (self.head_dim ** -0.5)
            )
            carried = select(window_notebook, over=window, index=length - 1)
        output = cat(reads, over="chunk_time", into=time)
        return output, carried

    def _chunked_fused(self, batch, time, queries, keys, values, decay):
        """Fused complex chunked read (raw-torch boundary; same math as
        ``_stable_notebook``).  See ``triton_kernels.fused_complex_pam_read``.
        """
        B, H, T, K = batch.size, self.heads.size, time.size, self.head_dim
        q = queries.raw(batch, self.heads, time, self.head_feature,
                        self.complex_pair).reshape(B * H, T, K, 2)
        k = keys.raw(batch, self.heads, time, self.head_feature,
                     self.complex_pair).reshape(B * H, T, K, 2)
        v = values.raw(batch, self.heads, time, self.head_feature,
                       self.complex_pair).reshape(B * H, T, K, 2)
        retention = decay.raw(batch, self.heads, time).reshape(B * H, T)
        read, state = fused_complex_pam_read(q, k, v, retention, None, self.chunk_size)
        output = named(read.reshape(B, H, T, K, 2) * (self.head_dim ** -0.5),
                       (batch, self.heads, time, self.head_row, self.complex_pair),
                       self.policy)
        carried = named(state.reshape(B, H, K, K, 2),
                        (batch, self.heads, self.head_row, self.head_col,
                         self.complex_pair), self.policy)
        return output, carried

    # ── decode: one step per token on the carried notebook ───────────────────

    def _stepwise(self, tokens: NamedTensor, queries, keys, values, decay,
                  state) -> tuple[NamedTensor, NamedTensor]:
        batch, time = tokens.layout[0], tokens.layout[1]
        seq_len = tokens.size(time)
        notebook = self._as_notebook(state, batch) if state is not None else zeros(
            batch, self.heads, self.head_row, self.head_col, self.complex_pair,
            policy=self.policy, device=tokens.device, dtype=tokens.dtype,
        )
        reads = []
        for t in range(seq_len):
            # The recurrence, one step: fade, write, then read.
            notebook = (select(decay, over=time, index=t) * notebook) + outer(
                select(values, over=time, index=t).alias(self.head_feature, self.head_row),
                select(keys, over=time, index=t).alias(self.head_feature, self.head_col).conj(),
                (self.head_row, self.head_col),
            )
            reads.append(
                contract(notebook,
                         select(queries, over=time, index=t)
                         .alias(self.head_feature, self.head_col),
                         over=self.head_col)
                * (self.head_dim ** -0.5)
            )
        output = stack(reads, into=time, at=self.head_row)
        return output, notebook

    def forward(self, x: NamedTensor, state=None, step_offset: int = 0):
        tokens = self._as_token(x)
        seq_len = tokens.size(tokens.layout[1])
        queries, keys, values = self._project(tokens, step_offset)
        decay = self._decay(tokens)

        if state is None and seq_len > 1:
            output, new_state = self._chunked(tokens, queries, keys, values, decay)
        else:
            output, new_state = self._stepwise(tokens, queries, keys, values, decay, state)

        out = self.o_proj(output.alias(self.head_row, self.head_feature)
                          .to(tokens.layout[0], tokens.layout[1], self.inner,
                              self.complex_pair))
        if self.training:
            out = out * as_complex_dropout_mask(self.out_dropout, out)
        return out, (new_state.data if new_state is not None else None)  # named-exit: state hand-off


# ── Block ────────────────────────────────────────────────────────────────────


class Block(nn.Module):
    """Pre-norm residual: gated channel mix, then PAM sequence mix."""

    def __init__(self, cfg: PAMConfig, layer_idx: int = 0,
                 model_dim: Dim | None = None, complex_pair: Dim | None = None):
        super().__init__()
        self.model_dim = model_dim or Dim("model_dim", cfg.dim)
        self.complex_pair = complex_pair or Dim("complex_pair", 2)
        self.dropout = cfg.dropout
        self.norm1 = ComplexNorm(self.model_dim, pair=self.complex_pair)
        self.cgu = ComplexGatedUnit(self.model_dim, cfg.expand,
                                    activation=cfg.activation,
                                    pair=self.complex_pair)
        self.cgu_dropout = nn.Dropout(cfg.dropout)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.norm2 = ComplexNorm(self.model_dim, pair=self.complex_pair)
        self.pam = PAMLayer(cfg, layer_idx=layer_idx,
                            model_dim=self.model_dim, complex_pair=self.complex_pair)
        # PAM starts soft: the memory path learns while the residual carries.
        self.pam_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: NamedTensor, pam_state=None, step_offset: int = 0):
        cgu_out = self.cgu(self.norm1(x))
        if self.training:
            cgu_out = cgu_out * as_complex_dropout_mask(self.cgu_dropout, cgu_out)
        x = x + cgu_out * self.cgu_scale
        pam_out, new_state = self.pam(self.norm2(x), state=pam_state,
                                      step_offset=step_offset)
        return x + pam_out * self.pam_scale, new_state


# ── Fully-real PAM ───────────────────────────────────────────────────────────


class RealPAMLayer(nn.Module):
    r"""One head-fan of phase-associative memories, in plain real arithmetic.

    Per head, a real d×d notebook ``S`` carries the whole past:

        S_t = decay_t * S_{t-1} + value_t (x) key_t
        y_t = d^{-1/2} * (S_t . query_t)

    The real twin of ``PAMLayer``: one GEMM per projection, no phase to
    manage, and no conjugate in the write (a later read with the same key
    still recalls the value — the read is a plain dot). ``decay_t`` is the
    only learned memory quantity — one number per head per token, read from
    the token's channels.

    ``forward`` returns the PAM output ``[batch, time, inner]`` (inner =
    heads × head_feature, folded back to model_dim by the output projection)
    and the notebook to carry on (raw ``[batch, heads, d, d]`` per ``state``;
    ``None`` when nothing was carried in and nothing should be returned).
    """

    def __init__(self, cfg: PAMConfig, layer_idx: int = 0,
                 model_dim: Dim | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.model_dim = model_dim or Dim("model_dim", cfg.dim)
        self.heads = Dim("heads", cfg.n_heads)
        self.head_feature = Dim("head_feature", cfg.head_dim)
        self.head_row = Dim("head_row", cfg.head_dim)
        self.head_col = Dim("head_col", cfg.head_dim)
        self.inner = ProductDim(self.heads, self.head_feature)
        # Real RoPE: one 2x2 rotation per channel pair.
        self.head_pair = Dim("head_pair", cfg.head_dim // 2)
        self.rot_pair = Dim("rot_pair", 2)

        # Fused QKV: one projection, then split. The three share the same
        # width, so one matrix is the whole fan.
        self.qkv_slot = Dim("qkv_slot", 3)
        self.qkv_fused = Dim("qkv_fused", 3 * cfg.n_heads * cfg.head_dim)
        self.qkv_proj = NamedLinear(self.model_dim, self.qkv_fused, bias=False)
        self.o_proj = NamedLinear(self.inner, self.model_dim, bias=False)

        # The decay: one number per head, read from the token's channels.
        self.decay_out = Dim("decay_out", cfg.n_heads)
        self.dt_proj = NamedLinear(self.model_dim, self.decay_out)
        self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads) + cfg.base_dt_bias)

        # RoPE table, built once outside the graph (declared boundary).
        if cfg.use_rope:
            self.register_buffer(
                'rope_cache',
                build_rope_cache_real(cfg.max_seq_len, cfg.head_dim // 2),
                persistent=False,
            )
        self.use_rope = cfg.use_rope

        self.chunk_size = cfg.chunk_size
        self.head_dim = cfg.head_dim
        self.out_dropout = nn.Dropout(cfg.dropout)
        # Diagnostic hook (off by default): see PAMLayer.capture_decay.
        self.capture_decay = False

        # A1: short depthwise causal conv on the fused qkv, added residually
        # and zero-init so it is the identity at start (Based / Gated-DeltaNet
        # convention; ~C*k params). Off unless cfg.short_conv.
        self.short_conv = cfg.short_conv
        if self.short_conv:
            self._conv_k = cfg.short_conv_k
            C = 3 * cfg.n_heads * cfg.head_dim
            self.qkv_conv = nn.Conv1d(C, C, kernel_size=self._conv_k, groups=C, bias=True)
            nn.init.zeros_(self.qkv_conv.weight)
            nn.init.zeros_(self.qkv_conv.bias)

        # A2: multiple independent PAM states per head (E3, real). Shared q/k/v;
        # per-state decay-logit offsets fan the retention time-constants; a
        # learned per-head mixing sums the reads. Off unless cfg.n_states > 1.
        self.n_states = cfg.n_states
        if self.n_states > 1:
            self.state_dim = Dim("pam_state", self.n_states)
            off = torch.linspace(-cfg.state_dt_spread, cfg.state_dt_spread, self.n_states)
            self.state_dt_offset = nn.Parameter(off)                       # [S]
            self.state_mix = nn.Parameter(
                torch.ones(cfg.n_heads, self.n_states) / self.n_states)
        # A2b: state 0 is a vault (retention pinned to 1) protected by a gate
        # p = sigmoid(Linear(x) - 3); g <- g(1-p)+p, v <- (1-p) v on the vault.
        self.vault = cfg.vault
        if self.vault:
            self.protect = NamedLinear(self.model_dim, self.decay_out)

        # A3: delta erase/write. Unit keys; per-head erase b_e=sigmoid(.-3) (cap
        # 0.95) and write b_w=sigmoid(.) gates. Off unless cfg.delta.
        self.delta = cfg.delta
        if self.delta:
            self.erase_proj = NamedLinear(self.model_dim, self.decay_out)
            self.write_proj = NamedLinear(self.model_dim, self.decay_out)

        # N1 Chrono-PAM: content-modulated rotary retention. A per-head time
        # warp g_t = exp(clamp(W x, +/-3)) multiplies the per-step RoPE angle;
        # the cumulative phase is cumsum_t(inv_freq * g_t). W is zero-init
        # (_zero_init marker below), so at start g=1 and the phase is exactly
        # pos*inv_freq == standard RoPE (parity test). Folds a complex rotating
        # retention into q,k -> the fused kernel is untouched. Off unless chrono.
        self.chrono = cfg.chrono
        if self.chrono:
            P = cfg.head_dim // 2
            self.register_buffer(
                'chrono_inv_freq',
                1.0 / (10000.0 ** (torch.arange(P).float() / P)),
                persistent=False,
            )
            self.warp_proj = NamedLinear(self.model_dim, self.decay_out)
            self.warp_proj._zero_init = True  # identity-init: g=1 == plain RoPE

        # N4: content-dependent per-head read-out gate, silu(W_g x + b_g) on
        # the memory read before o_proj. Zero-init W_g and b_g = silu^-1(1) so
        # the gate is exactly 1 at start (parity with the ungated model); the
        # model then learns per token/head how much of the read to let through
        # (the static pam_scale stays as the layer-wide scale). Off unless
        # cfg.out_gate.
        self.out_gate = cfg.out_gate
        if self.out_gate:
            self.gate_proj = NamedLinear(self.model_dim, self.decay_out)
            self.gate_proj._zero_init = True
            self.gate_proj._init_bias = 1.2785  # silu(1.2785) = 1.0000

    # ── small named helpers ──────────────────────────────────────────────────

    def _as_token(self, x: NamedTensor) -> NamedTensor:
        """Re-state the token axes under this layer's own Dim identities.

        Each forward wraps its batch/time in fresh Dims, and a carried
        notebook keeps the Dims of the call that created it; restating is
        an identity change only — no data moves.
        """
        batch, time = x.layout[0], x.layout[1]
        return named(x.data, (batch, time, self.model_dim))  # named-exit: restate

    def _as_notebook(self, state, batch: Dim) -> NamedTensor:
        """Re-state a carried notebook onto this call's axes (no data moves)."""
        layout = (batch, self.heads, self.head_row, self.head_col)
        data = state.data if isinstance(state, NamedTensor) else state  # named-exit: restate
        return named(data, layout)

    def _project(self, tokens: NamedTensor, step_offset: int, clock=None):
        """Query, key, value on (batch, heads, time, head_feature), + clock.

        The fused QKV axis splits with its factors at the end of the layout —
        the one order sempty's named split accepts — then heads move before
        time for the notebook. RoPE rotates query and key by their position
        (a 2x2 rotation per channel pair), so the notebook's associations
        carry word order. The 4th return is the Chrono clock carried out of
        this call (``None`` unless ``cfg.chrono``).
        """
        batch, time = tokens.layout[0], tokens.layout[1]
        qkv = self.qkv_proj(tokens)
        if self.short_conv:
            qkv = self._short_conv(qkv, batch, time)
        qkv = qkv.to(batch, time, self.qkv_slot, self.heads, self.head_feature)
        queries, keys, values = (qkv.select(self.qkv_slot, slot) for slot in (0, 1, 2))

        if self.use_rope and self.chrono:
            # N1: content-modulated rotary (learned time-warp). Replaces the
            # fixed RoPE rotation on q/k; identical to it at init. The clock
            # (per-head cumulative warp) is the position: carried in `clock`
            # across decode steps instead of `step_offset`.
            layout = (batch, self.heads, time, self.head_feature)
            queries, keys, clock_out = self._rotate_learned(
                queries, keys, tokens, batch, time, clock)
            return (queries.to(*layout), keys.to(*layout), values.to(*layout),
                    clock_out)

        if self.use_rope:
            position_end = step_offset + tokens.size(time)
            if position_end > self.rope_cache.shape[0]:
                self.register_buffer(
                    'rope_cache',
                    build_rope_cache_real(position_end * 2, self.head_dim // 2)
                        .to(tokens.device),
                    persistent=False,
                )
            rotation = named(
                self.rope_cache[step_offset:position_end].to(dtype=tokens.dtype),
                (time, self.head_pair, self.rot_pair),
            )  # named-exit: the RoPE table is a plain buffer, sliced here
            cos_t = rotation.select(self.rot_pair, 0)
            sin_t = rotation.select(self.rot_pair, 1)

            def _rotate(x):
                # Split each channel pair (even, odd) and rotate it by 2x2.
                split = x.to(batch, time, self.heads, self.head_pair, self.rot_pair)
                xs = split.select(self.rot_pair, 0)
                ys = split.select(self.rot_pair, 1)
                rx = xs * cos_t - ys * sin_t
                ry = xs * sin_t + ys * cos_t
                return stack([rx, ry], into=self.rot_pair).to(
                    batch, time, self.heads, self.head_feature)

            queries = _rotate(queries)
            keys = _rotate(keys)

        layout = (batch, self.heads, time, self.head_feature)
        return (queries.to(*layout), keys.to(*layout), values.to(*layout), None)

    def _short_conv(self, qkv: NamedTensor, batch, time) -> NamedTensor:
        """A1 depthwise causal conv on the fused qkv, residual + identity-init.

        Raw-torch boundary (Conv1d over the time axis): [B,T,C] -> [B,C,T],
        causal left-pad by k-1, depthwise conv, SiLU, back to [B,T,C], add.
        At init the conv weight/bias are zero so this is exactly the identity.
        """
        x = qkv.raw(batch, time, self.qkv_fused)              # [B, T, C]
        xc = x.transpose(1, 2)                                # [B, C, T]
        xc = nn.functional.pad(xc, (self._conv_k - 1, 0))
        conv = nn.functional.silu(self.qkv_conv(xc)).transpose(1, 2)
        return named(x + conv, (batch, time, self.qkv_fused))  # named-exit: conv boundary

    def _rotate_learned(self, queries, keys, tokens, batch, time, clock=None):
        """N1 Chrono-PAM rotary: content-modulated cumulative phase on q/k.

        Per-head time-warp ``g_t = exp(clamp(W x, +/-3))`` scales the per-step
        RoPE angle ``inv_freq``; the position phase is the *causal cumulative*
        sum ``Phi_t = sum_{j<t} inv_freq * g_j`` (exclusive, so ``Phi_0 = 0``).
        With ``W`` zero-init, ``g = 1`` and ``Phi_t = t * inv_freq`` -- exactly
        ``build_rope_cache_real`` (bit-parity with the baseline at start).

        This folds a complex rotating retention ``gamma_t = e^{i theta_t}`` into
        q/k, so the fused magnitude-retention kernel is untouched (speed held).
        Raw-torch boundary: cumsum over the time axis.

        ``clock`` is the per-head clock ``[B, H]`` (fp32) carried in from the
        tokens already processed (``None`` == 0, a fresh sequence); the
        returned ``clock_out`` is the clock after this call's tokens, so
        decode (``T=1`` steps) continues the same phase the prefill left off
        at. This is the Chrono analogue of ``step_offset``.
        """
        B, H, T, K = batch.size, self.heads.size, time.size, self.head_dim
        P = K // 2
        q = queries.raw(batch, self.heads, time, self.head_feature)   # [B,H,T,K]
        k = keys.raw(batch, self.heads, time, self.head_feature)
        # per-head warp in (~0.05, 20); zero-init W => exactly 1.0. Since inv is
        # constant in t, cumsum(inv*g) = inv * cumsum(g): warp a per-head clock
        # tau = cumsum(g) once ([B,H,T]) instead of a [B,H,T,P] cumsum.
        g = torch.exp(
            self.warp_proj(tokens).alias(self.decay_out, self.heads)
                .raw(batch, self.heads, time).float().clamp(-3.0, 3.0))  # [B,H,T] fp32
        run = torch.cumsum(g, dim=2)                                  # inclusive
        if clock is not None:
            run = run + clock.to(run.dtype).unsqueeze(-1)             # continue the clock
        tau = run - g                                                 # exclusive: tau_0=clock
        clock_out = run[:, :, -1]                                     # [B,H] after these tokens
        inv = self.chrono_inv_freq.to(device=q.device)               # [P] fp32
        phi = tau.unsqueeze(-1) * inv.view(1, 1, 1, P)                # [B,H,T,P] fp32
        # cos/sin fp32 for phase accuracy, cast to q dtype so the rotation and
        # the tensors autograd retains for backward stay in bf16 (memory/speed).
        cos = torch.cos(phi).to(q.dtype)
        sin = torch.sin(phi).to(q.dtype)                             # [B,H,T,P]
        qp = q.reshape(B, H, T, P, 2)
        kp = k.reshape(B, H, T, P, 2)
        qe, qo = qp.unbind(-1)
        ke, ko = kp.unbind(-1)
        qr = torch.stack((qe * cos - qo * sin, qe * sin + qo * cos), dim=-1)
        kr = torch.stack((ke * cos - ko * sin, ke * sin + ko * cos), dim=-1)
        layout = (batch, self.heads, time, self.head_feature)
        return (named(qr.reshape(B, H, T, K), layout),  # named-exit: learned-rotary boundary
                named(kr.reshape(B, H, T, K), layout),
                clock_out)

    def _decay(self, tokens: NamedTensor) -> NamedTensor:
        """Per-head retention in (0, 1): exp(-softplus(linear(token) + bias))."""
        batch, time = tokens.layout[0], tokens.layout[1]
        logit = self.dt_proj(tokens).alias(self.decay_out, self.heads)
        retention = exp(-softplus(logit + named(self.dt_bias, (self.heads,))))
        if self.capture_decay:
            _capture_retention(retention)
        return retention.to(batch, self.heads, time)

    # ── train / prefill: one closed form per window, notebook carried ────────

    def _chunked(self, tokens: NamedTensor, queries, keys, values,
                 decay) -> tuple[NamedTensor, NamedTensor]:
        """The window closed form, without the per-position notebook.

        ``notebook_s . q_s`` expands to ``a_s (S_in . q_s) + sum_{t<=s}
        M[s,t] (q_s . k_t) v_t`` — one ``[w, w]`` score matrix per window
        plus the carried ``[K, K]`` state, never the ``[w, K, K]`` notebook
        that ``_stable_notebook`` materialises (the complex arm still does).
        Same math, same carried state; ``triton_kernels.fused_real_pam_read``
        runs it fused (Triton fwd+bwd on CUDA) or in plain torch.  This
        method is the declared raw-torch boundary for that hand-off.
        """
        batch, time = tokens.layout[0], tokens.layout[1]
        B, H, T, K = batch.size, self.heads.size, time.size, self.head_dim
        q = queries.raw(batch, self.heads, time, self.head_feature).reshape(B * H, T, K)
        k = keys.raw(batch, self.heads, time, self.head_feature).reshape(B * H, T, K)
        v = values.raw(batch, self.heads, time, self.head_feature).reshape(B * H, T, K)
        retention = decay.raw(batch, self.heads, time).reshape(B * H, T)
        read, state = fused_real_pam_read(q, k, v, retention, None, self.chunk_size)
        output = named(read.reshape(B, H, T, K) * (self.head_dim ** -0.5),
                       (batch, self.heads, time, self.head_row))
        carried = named(state.reshape(B, H, K, K),
                        (batch, self.heads, self.head_row, self.head_col))
        return output, carried

    def _chunked_multi(self, tokens, queries, keys, values):
        """A2 multi-state chunked read (raw-torch boundary).

        Shared q/k/v; S states with per-state decay-logit offsets batched into
        the kernel's B*H*S axis; reads combined by a learned per-head mixing.
        A2b vault: state 0 retention pinned to 1, its writes gated by a protect
        gate p = sigmoid(Linear(x) - 3) (v_vault <- (1-p) v).
        """
        batch, time = tokens.layout[0], tokens.layout[1]
        B, H, T, K = batch.size, self.heads.size, time.size, self.head_dim
        S = self.n_states
        F_ = nn.functional
        q = queries.raw(batch, self.heads, time, self.head_feature)        # [B,H,T,K]
        k = keys.raw(batch, self.heads, time, self.head_feature)
        v = values.raw(batch, self.heads, time, self.head_feature)

        # per-state retention from the decay argument a = logit + dt_bias
        a = (self.dt_proj(tokens).alias(self.decay_out, self.heads)
             + named(self.dt_bias, (self.heads,)))
        a = a.raw(batch, self.heads, time)                                  # [B,H,T]
        offs = self.state_dt_offset.view(1, 1, S, 1)                        # [1,1,S,1]
        ret = torch.exp(-F_.softplus(a.unsqueeze(2) + offs))               # [B,H,S,T]
        v_s = v.unsqueeze(2).expand(B, H, S, T, K)                         # [B,H,S,T,K]

        if self.vault:
            # State 0 is permanent (retention 1) and its writes are gated by p;
            # rebuild slice 0 by concat (no in-place, keeps autograd happy).
            p = torch.sigmoid(
                self.protect(tokens).alias(self.decay_out, self.heads).raw(batch, self.heads, time)
                - 3.0)                                                      # [B,H,T]
            ret0 = torch.ones(B, H, 1, T, device=ret.device, dtype=ret.dtype)
            ret = torch.cat([ret0, ret[:, :, 1:]], dim=2)
            v0 = (v * (1.0 - p).unsqueeze(-1)).unsqueeze(2)                 # [B,H,1,T,K]
            v_s = torch.cat([v0, v_s[:, :, 1:]], dim=2)

        q_s = q.unsqueeze(2).expand(B, H, S, T, K).reshape(B * H * S, T, K)
        k_s = k.unsqueeze(2).expand(B, H, S, T, K).reshape(B * H * S, T, K)
        v_s = v_s.reshape(B * H * S, T, K)
        ret_s = ret.reshape(B * H * S, T)
        read, state = fused_real_pam_read(q_s, k_s, v_s, ret_s, None, self.chunk_size)
        read = read.reshape(B, H, S, T, K)
        alpha = self.state_mix.view(1, H, S, 1, 1)                          # [1,H,S,1,1]
        y = (read * alpha).sum(dim=2)                                       # [B,H,T,K]
        output = named(y * (self.head_dim ** -0.5),
                       (batch, self.heads, time, self.head_row))
        carried = named(state.reshape(B, H, S, K, K),
                        (batch, self.heads, self.state_dim, self.head_row, self.head_col))
        return output, carried

    def _chunked_delta(self, tokens, queries, keys, values, decay):
        """A3 delta erase/write chunked read (raw-torch boundary).

        Unit-norm keys; per-head erase/write gates; WY solve via
        ``triton_kernels.pam_delta_torch`` (which reuses the additive scan with
        pseudo-values).
        """
        batch, time = tokens.layout[0], tokens.layout[1]
        B, H, T, K = batch.size, self.heads.size, time.size, self.head_dim
        F_ = nn.functional
        q = queries.raw(batch, self.heads, time, self.head_feature).reshape(B * H, T, K)
        k = keys.raw(batch, self.heads, time, self.head_feature).reshape(B * H, T, K)
        k = F_.normalize(k, dim=-1)
        v = values.raw(batch, self.heads, time, self.head_feature).reshape(B * H, T, K)
        retention = decay.raw(batch, self.heads, time).reshape(B * H, T)
        be = torch.sigmoid(
            self.erase_proj(tokens).alias(self.decay_out, self.heads).raw(batch, self.heads, time)
            - 3.0).clamp(max=0.95).reshape(B * H, T)
        bw = torch.sigmoid(
            self.write_proj(tokens).alias(self.decay_out, self.heads).raw(batch, self.heads, time)
            ).reshape(B * H, T)
        read, state = pam_delta_torch(q, k, v, retention, bw, be, None, self.chunk_size)
        output = named(read.reshape(B, H, T, K) * (self.head_dim ** -0.5),
                       (batch, self.heads, time, self.head_row))
        carried = named(state.reshape(B, H, K, K),
                        (batch, self.heads, self.head_row, self.head_col))
        return output, carried

    # ── decode: one step per token on the carried notebook ───────────────────

    def _stepwise(self, tokens: NamedTensor, queries, keys, values, decay,
                  state) -> tuple[NamedTensor, NamedTensor]:
        batch, time = tokens.layout[0], tokens.layout[1]
        seq_len = tokens.size(time)
        notebook = self._as_notebook(state, batch) if state is not None else zeros(
            batch, self.heads, self.head_row, self.head_col,
            device=tokens.device, dtype=tokens.dtype,
        )
        reads = []
        for t in range(seq_len):
            # The recurrence, one step: fade, write, then read.
            notebook = (select(decay, over=time, index=t) * notebook) + outer(
                select(values, over=time, index=t).alias(self.head_feature, self.head_row),
                select(keys, over=time, index=t).alias(self.head_feature, self.head_col),
                (self.head_row, self.head_col),
            )
            reads.append(
                contract(notebook,
                         select(queries, over=time, index=t)
                         .alias(self.head_feature, self.head_col),
                         over=self.head_col)
                * (self.head_dim ** -0.5)
            )
        output = stack(reads, into=time, at=self.head_row)
        return output, notebook

    def forward(self, x: NamedTensor, state=None, step_offset: int = 0):
        tokens = self._as_token(x)
        seq_len = tokens.size(tokens.layout[1])
        # Chrono carries (notebook, clock); every other arm carries the notebook.
        if self.chrono and state is not None:
            state, clock = state
        else:
            clock = None
        queries, keys, values, clock_out = self._project(tokens, step_offset, clock)
        decay = self._decay(tokens)

        if state is None and seq_len > 1:
            if self.delta:
                output, new_state = self._chunked_delta(tokens, queries, keys, values, decay)
            elif self.n_states > 1:
                output, new_state = self._chunked_multi(tokens, queries, keys, values)
            else:
                output, new_state = self._chunked(tokens, queries, keys, values, decay)
        else:
            if self.n_states > 1 or self.delta:
                raise NotImplementedError(
                    "multi-state / delta decode is not implemented; run "
                    "rungs with --gen_every 0 (probe/val use the chunked path)")
            output, new_state = self._stepwise(tokens, queries, keys, values, decay, state)

        if self.out_gate:
            # N4: per-token, per-head read-out gate (broadcast over head_row).
            gate = silu(self.gate_proj(tokens).alias(self.decay_out, self.heads))
            output = output * gate.to(tokens.layout[0], self.heads, tokens.layout[1])

        out = self.o_proj(output.alias(self.head_row, self.head_feature)
                          .to(tokens.layout[0], tokens.layout[1], self.inner))
        if self.training:
            out = out * as_complex_dropout_mask(self.out_dropout, out)
        carried = new_state.data if new_state is not None else None  # named-exit: state hand-off
        if self.chrono:
            return out, (carried, clock_out)
        return out, carried


class RealBlock(nn.Module):
    """Pre-norm residual, real: gated channel mix, then real PAM sequence mix."""

    def __init__(self, cfg: PAMConfig, layer_idx: int = 0,
                 model_dim: Dim | None = None):
        super().__init__()
        self.model_dim = model_dim or Dim("model_dim", cfg.dim)
        self.dropout = cfg.dropout
        self.norm1 = RealNorm(self.model_dim)
        self.cgu = RealGatedUnit(self.model_dim, cfg.expand,
                                 activation=cfg.activation)
        self.cgu_dropout = nn.Dropout(cfg.dropout)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.norm2 = RealNorm(self.model_dim)
        self.pam = RealPAMLayer(cfg, layer_idx=layer_idx,
                                model_dim=self.model_dim)
        # PAM starts soft: the memory path learns while the residual carries.
        self.pam_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: NamedTensor, pam_state=None, step_offset: int = 0):
        cgu_out = self.cgu(self.norm1(x))
        if self.training:
            cgu_out = cgu_out * as_complex_dropout_mask(self.cgu_dropout, cgu_out)
        x = x + cgu_out * self.cgu_scale
        pam_out, new_state = self.pam(self.norm2(x), state=pam_state,
                                      step_offset=step_offset)
        return x + pam_out * self.pam_scale, new_state


# ── Language model ───────────────────────────────────────────────────────────


class LM(nn.Module):
    """Embed -> [Block] x N -> tied head, in one of two algebras.

    The head is tied to the embedding: the score of a candidate token is the
    dot product of its embedding with the hidden state — two real dot
    products in the complex model (``real . E_r + imag . E_i``), one in the
    fully-real one (``hidden . embed``). ``cfg.is_complex`` picks.
    """

    def __init__(self, cfg: PAMConfig):
        super().__init__()
        self.config = cfg
        self.model_dim = Dim("model_dim", cfg.dim)
        if cfg.is_complex:
            self.complex_pair = Dim("complex_pair", 2)
            self.real_imag_feature = Dim("real_imag_feature", cfg.dim * 2)
            self.policy = SplitComplex(self.complex_pair)
            self.embed = ComplexEmbed(cfg.vocab_size, self.model_dim, self.complex_pair)
            self.embed_norm = ComplexNorm(self.model_dim, pair=self.complex_pair)
            self.blocks = nn.ModuleList([
                Block(cfg, layer_idx=i,
                      model_dim=self.model_dim, complex_pair=self.complex_pair)
                for i in range(cfg.n_layers)
            ])
            self.output_norm = ComplexNorm(self.model_dim, pair=self.complex_pair)
            # A distinct out-axis: contract cannot tell two copies of one Dim apart.
            self.lm_head_out = Dim("lm_head_out", cfg.dim)
            self.lm_head_proj = ComplexLinear(self.model_dim, self.lm_head_out,
                                              pair=self.complex_pair)
            self.lm_head_norm = ComplexNorm(self.model_dim, pair=self.complex_pair)
        else:
            self.complex_pair = None
            self.real_imag_feature = None
            self.policy = None
            self.embed = RealEmbed(cfg.vocab_size, self.model_dim)
            self.embed_norm = RealNorm(self.model_dim)
            self.blocks = nn.ModuleList([
                RealBlock(cfg, layer_idx=i, model_dim=self.model_dim)
                for i in range(cfg.n_layers)
            ])
            self.output_norm = RealNorm(self.model_dim)
            # A distinct out-axis: contract cannot tell two copies of one Dim apart.
            self.lm_head_out = Dim("lm_head_out", cfg.dim)
            self.lm_head_proj = NamedLinear(self.model_dim, self.lm_head_out)
            self.lm_head_norm = RealNorm(self.model_dim)
        # A4: conditional n-gram memory after selected blocks (real arm only).
        self.cond_mem_layers = tuple(cfg.cond_mem_layers) if cfg.cond_mem else ()
        if self.cond_mem_layers:
            assert not cfg.is_complex, "cond_mem is implemented for the real arm"
            from v13_sempty.cond_mem import ConditionalMemory
            self.cond_mem = nn.ModuleDict({
                str(i): ConditionalMemory(cfg, self.model_dim)
                for i in self.cond_mem_layers
            })
        self._init_weights()

    def _init_weights(self):
        if self.config.is_complex:
            embed_weights = {self.embed.embed_real, self.embed.embed_imag}
        else:
            embed_weights = {self.embed.embed}
        for module in self.modules():
            if isinstance(module, (nn.Linear, NamedLinear)):
                if getattr(module, '_zero_init', False):
                    # Identity-init projections (Chrono warp, N4 gate): zero
                    # weight; bias = the constant that makes the layer a no-op
                    # at start (0 for the warp, silu^-1(1) for the gate).
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, getattr(module, '_init_bias', 0.0))
                    continue
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module not in embed_weights:
                nn.init.normal_(module.weight, std=0.02)

    def _stem(self, input_ids, step_offset: int = 0) -> tuple:
        batch, time = Dim("batch", input_ids.shape[0]), Dim("time", input_ids.shape[1])
        return self.embed_norm(self.embed(input_ids, batch, time)), batch, time

    def _tied_logits(self, lm: NamedTensor, batch: Dim, time: Dim) -> torch.Tensor:
        if self.config.is_complex:
            embed_real = named(self.embed.embed_real.weight, (self.embed.vocab, self.model_dim))
            embed_imag = named(self.embed.embed_imag.weight, (self.embed.vocab, self.model_dim))
            logits = (contract(real(lm), embed_real, over=self.model_dim)
                      + contract(imag(lm), embed_imag, over=self.model_dim))
        else:
            embed = named(self.embed.embed.weight, (self.embed.vocab, self.model_dim))
            logits = contract(lm, embed, over=self.model_dim)
        return logits.raw(batch, time, self.embed.vocab)  # named-exit: public logits are raw

    def _run_blocks(self, z, batch, time, states, step_offset, input_ids=None):
        use_ckpt = self.config.gradient_checkpointing and self.training and states is None
        new_states = []
        for i, block in enumerate(self.blocks):
            state = states[i] if states is not None else None
            if use_ckpt:
                z, new_state = self._ckpt_block(block, z, step_offset)
            else:
                z, new_state = block(z, pam_state=state, step_offset=step_offset)
            new_states.append(new_state)
            if self.cond_mem_layers and i in self.cond_mem_layers:
                z = self.cond_mem[str(i)](z, input_ids)
        return z, new_states

    def forward(self, input_ids, states=None, step_offset: int = 0, labels=None):
        z, batch, time = self._stem(input_ids, step_offset)
        z, new_states = self._run_blocks(z, batch, time, states, step_offset, input_ids)
        lm = self.lm_head_norm(self.lm_head_proj(self.output_norm(z)))
        logits = self._tied_logits(lm, batch, time)
        aux_loss = torch.zeros((), device=input_ids.device)
        return logits, new_states, aux_loss

    def _hidden_to_lm(self, input_ids, step_offset: int = 0):
        """Training path: the stack, stopped before full [B, T, V] logits."""
        z, batch, time = self._stem(input_ids, step_offset)
        z, _ = self._run_blocks(z, batch, time, states=None, step_offset=step_offset,
                                input_ids=input_ids)
        lm = self.lm_head_norm(self.lm_head_proj(self.output_norm(z)))
        aux_loss = torch.zeros((), device=input_ids.device)
        return lm, aux_loss

    def ce_from_lm(self, lm: NamedTensor, labels, loss_mask=None,
                   ignore_index=-100, chunk: int = 4096, return_nll: bool = False,
                   gemm_dtype=None):
        """Chunked cross-entropy from the pre-logit hidden ``lm``.

        The tied head folds into one real matmul, ``H @ W^T``. In the complex
        model ``H = concat(lm_real, lm_imag)`` and ``W = concat(embed_real,
        embed_imag)``; in the real model ``H`` is ``lm`` and ``W`` is the
        single embedding. Either way the chunked-CE Function never holds the
        full ``[N, vocab]`` softmax. Named up to the Function hand-off.
        ``gemm_dtype`` (None = autocast dtype, ``torch.float32`` = exact) is
        the head-GEMM precision; the loss itself is always fp32.
        """
        from v13_sempty.fused_ce import fused_linear_cross_entropy
        batch, time = lm.layout[0], lm.layout[1]
        flat = Dim("flat", batch.size * time.size)

        if self.config.is_complex:
            hidden = flatten(to_real_concat(lm, into=self.real_imag_feature),
                             (batch, time), flat)
            embed_real = named(self.embed.embed_real.weight, (self.embed.vocab, self.model_dim))
            embed_imag = named(self.embed.embed_imag.weight, (self.embed.vocab, self.model_dim))
            weight = cat([embed_real, embed_imag], over=self.model_dim,
                         into=self.real_imag_feature)
            feature = self.real_imag_feature
        else:
            hidden = flatten(lm, (batch, time), flat)
            weight = named(self.embed.embed.weight, (self.embed.vocab, self.model_dim))
            feature = self.model_dim

        out = fused_linear_cross_entropy(
            hidden.raw(flat, feature),  # named-exit: the CE Function takes raw torch
            weight.raw(self.embed.vocab, feature),  # named-exit: ditto
            labels.reshape(-1),  # named-exit: raw int tensor from the dataloader
            mask=(loss_mask.reshape(-1) if loss_mask is not None else None),  # named-exit: raw mask
            chunk=chunk, ignore_index=ignore_index, return_nll=return_nll,
            gemm_dtype=gemm_dtype,
        )
        if return_nll:
            return out, getattr(out, '_nll', None).reshape(batch.size, time.size)  # named-exit: raw nll
        return out

    def fused_ce_loss(self, input_ids, labels, loss_mask=None, ignore_index=-100,
                      chunk: int = 4096):
        """Eager convenience: the hidden stack plus chunked CE in one call."""
        lm, aux_loss = self._hidden_to_lm(input_ids)
        main = self.ce_from_lm(lm, labels, loss_mask=loss_mask,
                               ignore_index=ignore_index, chunk=chunk)
        return main, aux_loss

    @staticmethod
    def _ckpt_block(block, z, step_offset):
        # Non-reentrant checkpoint: backward recomputes the block with grad.
        # Do NOT detach the input — the checkpoint needs the input edge to
        # push gradient back through the preceding block.
        compiled = getattr(block, '_compiled_fwd', None)

        def run(z_in):
            if compiled is not None:
                return compiled(z_in, pam_state=None, step_offset=step_offset)
            return block(z_in, pam_state=None, step_offset=step_offset)

        return grad_checkpoint(run, z, use_reentrant=False)

    def compile_blocks(self, mode: str = 'default'):
        """Compile each Block.forward so inductor can see the PAM chunk."""
        for block in self.blocks:
            block._compiled_fwd = torch.compile(block.forward, mode=mode, dynamic=False)
        return self

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0,
                 top_k=50, top_p=0.0, repetition_penalty=1.0, eos_token_id=None):
        """Autoregressive decode: prefill builds the notebooks, then O(1)/token."""
        self.eval()
        generated = input_ids.clone()
        logits, states, _ = self.forward(generated)
        step = generated.shape[1]
        finished = torch.zeros(generated.shape[0], dtype=torch.bool,
                               device=generated.device)
        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                # Greedy: the argmax is the whole distribution.
                next_logits = logits[:, -1]
                if repetition_penalty != 1.0:
                    score = torch.gather(next_logits, 1, generated)
                    score = torch.where(score > 0, score / repetition_penalty,
                                        score * repetition_penalty)
                    next_logits.scatter_(1, generated, score)
                nxt = next_logits.argmax(-1, keepdim=True)
            else:
                next_logits = logits[:, -1] / temperature
                if repetition_penalty != 1.0:
                    score = torch.gather(next_logits, 1, generated)
                    score = torch.where(score > 0, score / repetition_penalty,
                                        score * repetition_penalty)
                    next_logits.scatter_(1, generated, score)
                if top_k > 0:
                    v, _ = next_logits.topk(min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, -1:]] = float('-inf')
                if top_p > 0:
                    sl, si = next_logits.sort(descending=True)
                    cum = sl.softmax(dim=-1).cumsum(dim=-1)
                    sl[cum - sl.softmax(dim=-1) >= top_p] = float('-inf')
                    next_logits = sl.scatter(1, si, sl)
                nxt = torch.multinomial(next_logits.softmax(dim=-1), 1)
            generated = torch.cat([generated, nxt], dim=1)
            if eos_token_id is not None:
                finished |= nxt.squeeze(1) == eos_token_id
                if bool(finished.all()):
                    break
            logits, states, _ = self.forward(nxt, states=states, step_offset=step)
            step += 1
        return generated

    def cond_mem_table_param_ids(self) -> set:
        """ids of the A4 lookup-table weights (reported/optimised separately)."""
        if not self.cond_mem_layers:
            return set()
        return {id(m.table.weight) for m in self.cond_mem.values()}

    def count_parameters(self) -> Dict[str, int]:
        embed_p = sum(p.numel() for p in self.embed.parameters())
        block_p = sum(p.numel() for b in self.blocks for p in b.parameters())
        head_p = (sum(p.numel() for p in self.lm_head_proj.parameters())
                  + sum(p.numel() for p in self.lm_head_norm.parameters()))
        norm_p = (sum(p.numel() for p in self.embed_norm.parameters())
                  + sum(p.numel() for p in self.output_norm.parameters()))
        cm_table_p, cm_dense_p = 0, 0
        if self.cond_mem_layers:
            table_ids = self.cond_mem_table_param_ids()
            for m in self.cond_mem.values():
                for p in m.parameters():
                    if id(p) in table_ids:
                        cm_table_p += p.numel()
                    else:
                        cm_dense_p += p.numel()
        # 'total' = dense params (comparison budget); table is reported apart.
        total = embed_p + block_p + head_p + norm_p + cm_dense_p
        out = {'embedding (tied)': embed_p, 'blocks': block_p,
               'norms': norm_p, 'lm_head': head_p, 'total': total}
        if self.cond_mem_layers:
            out['cond_mem_dense'] = cm_dense_p
            out['cond_mem_table'] = cm_table_p
            out['total_with_table'] = total + cm_table_p
        return out


__all__ = ["PAMLayer", "RealPAMLayer", "Block", "RealBlock", "LM"]
