r"""v14 units on the v13_sempty body.

N1 — phase interference between notebooks (v11 E3, rebuilt on the lean real
body with delta writes). Per head there are K delta notebooks that share the
query, key, value and write/erase gates; only the fade differs (one decay
offset per notebook). Each notebook is read with the same query, then the K
reads are combined by one of three rules (``V14Config.phase_read``):

    sum    y = Σ_k a_{h,k} · y_k                 a: learned per head, a = 1/K at start
    gate   y = (1/K) Σ_k (1 + c_k(x)) · y_k       real data-dependent mix
    phase  y = (1/K) Σ_k R(φ_k(x)) · y_k          rotation interference

``φ_k(x) = W x`` and ``c_k(x) = W x`` (one number per head and notebook, W
zero-init); ``R(φ)`` turns every channel pair of the read by φ, i.e.
multiplies it by e^{iφ}. At start both are 0, so all three rules are the
plain average of the K reads. ``gate`` and ``phase`` have the same
parameters; ``gate`` scales each read by a real number (it may go negative),
``phase`` turns it. The pair isolates whether turning carries anything that a
real data-dependent weight does not. (A ``cos φ`` gate was rejected: its
gradient at φ = 0 is zero, so it could never leave the start.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sempyt import (cat, exp, mean, rotate, scan, select, shift, sigmoid, softplus,
                    sqrt, square, stack, sum as named_sum, take)
from sempyt.dim import Dim, ProductDim
from sempyt.policies import REAL, SplitComplex
from sempyt.nn import Linear as NamedLinear
from sempyt.tensor import NamedTensor, named

from v13_sempty.model import LM, RealPAMLayer
from v13_sempty.triton_kernels import pam_delta_batched

from v14.config import V14Config

PHASE_READS = ('sum', 'gate', 'phase')
OPMEM_MODES = ('phase', 'diag')


class PhaseStatesPAM(RealPAMLayer):
    """N1: K delta notebooks per head, combined by sum / gate / phase."""

    def __init__(self, cfg: V14Config, layer_idx: int = 0, model_dim: Dim | None = None):
        super().__init__(cfg, layer_idx=layer_idx, model_dim=model_dim)
        if not cfg.delta:
            raise ValueError("N1 phase_states needs delta=True")
        if cfg.share_reads:
            raise ValueError("N1 phase_states does not support M2 share_reads")
        if cfg.phase_read not in PHASE_READS:
            raise ValueError(f"phase_read must be one of {PHASE_READS}")
        self.n_notebooks = cfg.phase_states
        self.phase_read = cfg.phase_read
        self.notebook = Dim("notebook", self.n_notebooks)
        spread = cfg.phase_dt_spread
        self.notebook_dt_offset = nn.Parameter(
            torch.linspace(-spread, spread, self.n_notebooks))
        if self.phase_read == 'sum':
            self.notebook_mix = nn.Parameter(torch.ones(cfg.n_heads, self.n_notebooks) / self.n_notebooks)
        else:
            self.angle_out = ProductDim(self.heads, self.notebook)
            self.angle_proj = NamedLinear(self.model_dim, self.angle_out)
            self.angle_proj._zero_init = True          # φ = 0 at start

    # ── combine K reads (named) ──────────────────────────────────────────────

    def _combine(self, tokens: NamedTensor, reads: NamedTensor) -> NamedTensor:
        """reads (batch, heads, notebook, time, head_row) -> (batch, heads, time, head_row)."""
        batch, time = tokens.layout[0], tokens.layout[1]
        if self.phase_read == 'sum':
            mix = named(self.notebook_mix, (self.heads, self.notebook))
            return named_sum(reads * mix, over=self.notebook).to(
                batch, self.heads, time, self.head_row)
        angle = self.angle_proj(tokens).to(batch, self.heads, self.notebook, time)
        if self.phase_read == 'gate':
            turned = reads * (angle + 1.0)
        else:
            planes = reads.to(batch, self.heads, self.notebook, time,
                              self.head_pair, self.rot_pair)
            turned = rotate(planes, angle, self.rot_pair).to(
                batch, self.heads, self.notebook, time, self.head_row)
        return (named_sum(turned, over=self.notebook) * (1.0 / self.n_notebooks)).to(
            batch, self.heads, time, self.head_row)

    # ── per-notebook fade ────────────────────────────────────────────────────

    def _notebook_retention(self, tokens: NamedTensor) -> torch.Tensor:
        """Raw [B, H, K, T] retention exp(-softplus(a + offset_k))."""
        batch, time = tokens.layout[0], tokens.layout[1]
        a = (self.dt_proj(tokens).alias(self.decay_out, self.heads)
             + named(self.dt_bias, (self.heads,))).raw(batch, self.heads, time)
        offset = self.notebook_dt_offset.view(1, 1, -1, 1)
        return torch.exp(-nn.functional.softplus(a.unsqueeze(2) + offset))

    # ── train / prefill: K notebooks batched into the delta kernel ──────────

    def _chunked_delta(self, tokens, queries, keys, values, decay):
        """Raw-torch boundary: K notebooks share q/k/v/gates, batched on B*H*K."""
        batch, time = tokens.layout[0], tokens.layout[1]
        B, H, T, D = batch.size, self.heads.size, time.size, self.head_dim
        K = self.n_notebooks
        q = queries.raw(batch, self.heads, time, self.head_feature)
        k = nn.functional.normalize(
            keys.raw(batch, self.heads, time, self.head_feature), dim=-1)
        v = values.raw(batch, self.heads, time, self.head_feature)
        ret = self._notebook_retention(tokens)                         # [B,H,K,T]
        bw, be = self._delta_gates(tokens)                             # [B,H,T]
        fan = self._fan
        read, state = pam_delta_batched(
            fan(q), fan(k), fan(v), ret.reshape(B * H * K, T), fan(bw), fan(be), None)
        reads = named(read.reshape(B, H, K, T, D) * (D ** -0.5),
                      (batch, self.heads, self.notebook, time, self.head_row))
        carried = named(state.reshape(B, H, K, D, D),
                        (batch, self.heads, self.notebook, self.head_row, self.head_col))
        return self._combine(tokens, reads), carried

    def _fan(self, x: torch.Tensor) -> torch.Tensor:
        """Raw [B, H, ...] -> [B*H*K, ...]: the same tensor for every notebook."""
        B, H, K = x.shape[0], x.shape[1], self.n_notebooks
        return x.unsqueeze(2).expand(B, H, K, *x.shape[2:]).reshape(B * H * K, *x.shape[2:])

    # ── decode: one erase/write step per token on K carried notebooks ───────

    def _stepwise_delta(self, tokens, queries, keys, values, decay, state):
        """Raw-torch boundary: the K-notebook recurrence, one token at a time.

            S_t^k = g_t^k S_{t-1}^k (I - b_e k_t k_tᵀ) + b_w v_t k_tᵀ ,   y_t^k = S_t^k q_t
        """
        batch, time = tokens.layout[0], tokens.layout[1]
        B, H, T, D = batch.size, self.heads.size, time.size, self.head_dim
        K = self.n_notebooks
        q = queries.raw(batch, self.heads, time, self.head_feature).float()
        k = nn.functional.normalize(
            keys.raw(batch, self.heads, time, self.head_feature).float(), dim=-1)
        v = values.raw(batch, self.heads, time, self.head_feature).float()
        g = self._notebook_retention(tokens).float()                   # [B,H,K,T]
        bw, be = (x.float() for x in self._delta_gates(tokens))       # [B,H,T]
        if state is not None:
            S = (state.data if isinstance(state, NamedTensor) else state).float()  # named-exit: restate
        else:
            S = torch.zeros(B, H, K, D, D, device=tokens.device, dtype=torch.float32)
        reads = []
        for t in range(T):
            kt = k[:, :, None, t]                                       # [B,H,1,D]
            Sk = torch.einsum('bhnvk,bhnk->bhnv', S, kt.expand(B, H, K, D))
            erased = S - be[:, :, t, None, None, None] * Sk.unsqueeze(-1) * kt.unsqueeze(-2)
            S = (g[:, :, :, t, None, None] * erased
                 + bw[:, :, t, None, None, None]
                 * v[:, :, None, t].unsqueeze(-1) * kt.unsqueeze(-2))
            reads.append(torch.einsum('bhnvk,bhk->bhnv', S, q[:, :, t]))
        y = torch.stack(reads, dim=3).to(tokens.dtype) * (D ** -0.5)  # [B,H,K,T,D]
        reads_named = named(y, (batch, self.heads, self.notebook, time, self.head_row))
        carried = named(S, (batch, self.heads, self.notebook, self.head_row, self.head_col))
        return self._combine(tokens, reads_named), carried


class OperatorMemory(nn.Module):
    r"""N6: a memory of how things change, applied to the next input.

    Each channel pair of each head is one complex number. At every token t:

        a_t = W_in h_{t-1}            the input side   (what came before)
        b_t = W_role ê(x_t)           the output side  (the token that followed)
        r_t = b_t · conj(a_t)         their relation: a rotation (and a scale)
        S_t = γ_t S_{t-1} + w_t r_t   relations accumulated, faded per head
        q_t = W_in h_t                the next input
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
        self.mode = cfg.opmem
        self.model_dim = model_dim
        self.heads = Dim("op_heads", cfg.n_heads)
        self.plane = Dim("op_plane", cfg.head_dim // 2)
        self.pair = Dim("op_pair", 2)
        self.feat = ProductDim(self.heads, self.plane, self.pair)
        self.w_in = NamedLinear(model_dim, self.feat, bias=False)
        self.w_role = NamedLinear(model_dim, self.feat, bias=False)
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
        a = self._complex(self.w_in(h_prev), batch, time)
        b = self._complex(self.w_role(unit_embed), batch, time)
        q = self._complex(self.w_in(hidden), batch, time)
        if self.mode == 'phase':
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
        if self.mode == 'phase':
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


class V14LM(LM):
    """The v13_sempty LM with v14 units swapped in. ``V14Config()`` == v13 LM."""

    def __init__(self, cfg: V14Config):
        super().__init__(cfg)
        swapped = False
        self.opmem = None
        if getattr(cfg, 'opmem', ''):
            if cfg.is_complex or not cfg.echo:
                raise ValueError("N6 opmem is implemented on the real echo body")
            self.opmem = OperatorMemory(cfg, self.model_dim)
            swapped = True
        if getattr(cfg, 'phase_states', 1) > 1:
            if cfg.is_complex:
                raise ValueError("N1 is implemented for the real arm")
            for i, block in enumerate(self.blocks):
                if block.pam is not None:
                    block.pam = PhaseStatesPAM(cfg, layer_idx=i, model_dim=self.model_dim)
                    swapped = True
        if swapped:
            # Re-run the body's init so the new layers follow the same rules
            # (normal 0.02, zero-init markers); then restore echo's fade bias.
            self._init_weights()
            if self.echo:
                with torch.no_grad():
                    self.echo_gates.bias[:cfg.n_heads].fill_(cfg.echo_dt_bias)

    def _lm_vector(self, z, input_ids, batch, time, echo_state=None):
        """Pre-logit vector: head + echo (as in v13) + N6 operator memory.
        With opmem the echo state slot carries (echo_state, op_state)."""
        if self.opmem is None:
            return super()._lm_vector(z, input_ids, batch, time, echo_state)
        e_state, o_state = echo_state if echo_state is not None else (None, None)
        hidden = self.output_norm(z)
        lm = self.lm_head_norm(self.lm_head_proj(hidden))
        out, new_echo = self._echo_read(hidden, input_ids, batch, time, e_state)
        lm = lm + out
        op_out, new_op = self.opmem(hidden, self.embed(input_ids, batch, time), o_state)
        return lm + op_out, (new_echo, new_op)
