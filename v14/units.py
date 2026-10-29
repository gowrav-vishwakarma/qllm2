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

Tied echo (``V14Config.echo_tied_qk``): echo with one projection for query
and key (W_q = W_k), so a query lands on keys of hidden states like its own.
The B1 run 2 lead (label_category 0.857 vs echo 0.471, one seed).

N6 operator memory was removed after B1 run 2 (FAIL); its code is at
``78e4354`` (v14/EXPERIMENT_INDEX.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sempyt import rotate, sum as named_sum
from sempyt.dim import Dim, ProductDim
from sempyt.nn import Linear as NamedLinear
from sempyt.tensor import NamedTensor, named

from v13_sempty.model import LM, RealPAMLayer
from v13_sempty.triton_kernels import pam_delta_batched

from v14.config import V14Config

PHASE_READS = ('sum', 'gate', 'phase')


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


class TiedEchoQK(nn.Module):
    """Echo's query/key projection with W_q = W_k: one D x D map whose output
    is returned twice, so v13's ``q, k = echo_qk(h).chunk(2)`` gives q = k."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, h):
        y = self.proj(h)
        return torch.cat([y, y], dim=-1)  # named-exit: v13 echo boundary (raw torch)


class V14LM(LM):
    """The v13_sempty LM with v14 units swapped in. ``V14Config()`` == v13 LM."""

    def __init__(self, cfg: V14Config):
        super().__init__(cfg)
        swapped = False
        if getattr(cfg, 'echo_tied_qk', False):
            if not cfg.echo:
                raise ValueError("echo_tied_qk needs echo")
            self.echo_qk = TiedEchoQK(cfg.dim)
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
