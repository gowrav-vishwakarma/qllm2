r"""MPO-Tensorized Effect Bank (post-survival v0 stub).

This is a **research stub** for the tensor-network follow-on described in
:doc:`MPO_BANK.md`. It is intentionally minimal:

* Real-valued cores only (split-real tensors carry an explicit zero imag part
  to stay shape-compatible with the rest of v8).
* Effect *directions* and *values* live in an MPO with bond dimension
  ``bond_dim`` over ``n_cores`` cores, total effective bank size
  ``M_eff = prod(K_per_core)``.
* The probe step **materializes** the full ``M_eff`` direction tensor in v0.
  This is *not* the production path -- the point is to test whether the MPO
  factorization is a useful inductive bias relative to a flat
  :class:`~v8.qlc.effect_bank.EffectAlgebraBank`. A sublinear top-k contraction
  (TN-style) is a TODO for v1.
* Same call signature as
  :meth:`~v8.qlc.effect_bank.EffectAlgebraBank.select_top_k`, so this can be
  dropped into :class:`~v8.qlc.reason_loop.QuantumLogicCore` once the
  discriminator suite earns it.

This module is **not** wired into any preset and **not** used by Stage A or
Stage B. It exists so the design is reviewable and the API is locked in.

See :doc:`MPO_BANK.md` for math and experimental plan.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _factorize(dim: int, n_cores: int) -> List[int]:
    """Pick a length-``n_cores`` factorization of ``dim`` with roughly equal
    factors. Falls back to ``[dim, 1, 1, ...]`` if no clean factorization
    exists. Used to map ``psi: [B, d]`` onto the MPO core layout.
    """
    if n_cores <= 1:
        return [dim]
    target = round(dim ** (1.0 / n_cores))
    factors: List[int] = []
    remaining = dim
    for i in range(n_cores - 1):
        f = max(1, target)
        while f > 1 and remaining % f != 0:
            f -= 1
        factors.append(f)
        remaining //= f
    factors.append(remaining)
    if math.prod(factors) != dim:
        return [dim] + [1] * (n_cores - 1)
    return factors


class MPOEffectBank(nn.Module):
    r"""Effect bank whose ``M_eff`` rank-1 effects live in an MPO over
    ``n_cores`` cores with bond dimension ``bond_dim``.

    API mirror of :class:`~v8.qlc.effect_bank.EffectAlgebraBank`:

    * :meth:`probe` returns ``[B, H, M_eff]`` scores.
    * :meth:`select_top_k` returns ``(U, V)`` tensors with the same shapes
      that the SPM consumes via
      :meth:`~v8.qlc.projector.SasakiProjectionMemory.build_from_basis`.

    The ``infonce_loss`` auxiliary is **not** implemented in v0; if/when this
    bank graduates to production, it can borrow the implementation from
    :meth:`~v8.qlc.effect_bank.EffectAlgebraBank.infonce_loss` verbatim
    (the contrastive loss only depends on probe outputs).

    Parameters
    ----------
    dim
        State dimension :math:`d`. Must factor exactly into ``n_cores``
        per-core spatial dims (we attempt automatic factorization).
    bank_size_per_core
        Number of *effect-index* values per core (``K_c``). Total effective
        bank size is ``M_eff = K^n_cores``.
    n_cores
        Number of MPO cores ``C``.
    bond_dim
        Bond dimension :math:`\\chi`. Larger = more expressive.
    n_heads
        Number of reasoning heads sharing the bank (matches
        ``EffectAlgebraBank.n_heads`` semantics).
    """

    def __init__(
        self,
        dim: int,
        bank_size_per_core: int,
        n_cores: int = 3,
        bond_dim: int = 4,
        n_heads: int = 1,
        init_scale: float = 0.02,
        gate_init: float = -2.0,
        max_materialized_effects: int = 16384,
    ):
        super().__init__()
        self.dim = dim
        self.K = bank_size_per_core
        self.C = n_cores
        self.chi = bond_dim
        self.n_heads = n_heads
        self.max_materialized_effects = max_materialized_effects

        d_per_core = _factorize(dim, n_cores)
        if math.prod(d_per_core) != dim:
            raise ValueError(
                f"dim={dim} could not be factored into {n_cores} cores; "
                f"choose a dim with a non-trivial factorization."
            )
        self.d_per_core: Tuple[int, ...] = tuple(d_per_core)

        m_eff = self.K ** self.C
        if m_eff > max_materialized_effects:
            raise ValueError(
                f"M_eff={m_eff} exceeds max_materialized_effects="
                f"{max_materialized_effects}. v0 stub requires a small "
                f"effective bank; sublinear top-k is a TODO."
            )
        self.bank_size = m_eff

        # MPO cores for effect directions.
        # Shape per core: [chi_left, K, d_c, chi_right]; left/right boundary
        # bonds are 1.
        self.dir_cores = nn.ParameterList()
        for c, d_c in enumerate(self.d_per_core):
            left = 1 if c == 0 else self.chi
            right = 1 if c == self.C - 1 else self.chi
            core = nn.Parameter(torch.randn(left, self.K, d_c, right) * init_scale)
            self.dir_cores.append(core)

        # MPO cores for value vectors (independent factorization).
        self.val_cores = nn.ParameterList()
        for c, d_c in enumerate(self.d_per_core):
            left = 1 if c == 0 else self.chi
            right = 1 if c == self.C - 1 else self.chi
            core = nn.Parameter(torch.randn(left, self.K, d_c, right) * init_scale)
            self.val_cores.append(core)

        # Effect gates as a flat tensor over M_eff. v1 should factor these
        # over cores too (e.g. additive log-gates per core).
        self.s_logit = nn.Parameter(torch.full((m_eff,), gate_init))

        # Per-head probe transform (same shape as flat bank for consistency).
        self.probe_w = nn.Parameter(torch.empty(n_heads, dim, dim))
        for h in range(n_heads):
            nn.init.orthogonal_(self.probe_w[h], gain=(1.0 / math.sqrt(dim)))

    # ── Effect tensor materialization ──────────────────────────────────────

    def _contract_cores(self, cores: nn.ParameterList) -> torch.Tensor:
        """Contract MPO cores into a dense ``[M_eff, d]`` real matrix.

        v0 only: enumerates all bond-index sequences. Cost is
        ``O(M_eff * d * chi)`` -- usable for small banks, prohibitive for
        large ones. v1 should keep cores separate and contract only at the
        selected (m₁, …, m_C) tuples.
        """
        # Iterate cores and build a tensor of shape [M_eff_so_far, d_so_far, chi_right]
        cur: torch.Tensor = cores[0].squeeze(0)        # [K, d_0, chi]
        cur = cur.reshape(self.K, self.d_per_core[0], -1)

        for c in range(1, self.C):
            nxt = cores[c]                              # [chi, K, d_c, chi_right]
            # Tensor product over the m-index, contraction over the bond:
            # cur: [M_so_far, d_so_far, chi] x nxt: [chi, K, d_c, chi_right]
            #   -> [M_so_far, d_so_far, K, d_c, chi_right]
            #   -> [M_so_far*K, d_so_far*d_c, chi_right]
            m_so_far, d_so_far, _ = cur.shape
            cur = torch.einsum("mda,akeb->mdkeb", cur, nxt)
            cur = cur.permute(0, 2, 1, 3, 4).reshape(
                m_so_far * self.K,
                d_so_far * self.d_per_core[c],
                -1,
            )

        # cur shape: [M_eff, d, 1] -> [M_eff, d]
        return cur.squeeze(-1)

    def effect_directions(self) -> torch.Tensor:
        """Return ``[M_eff, d, 2]`` unit-norm direction vectors (split-real,
        imaginary part zero in v0)."""
        u_real = self._contract_cores(self.dir_cores)            # [M_eff, d]
        # Normalize each direction.
        u_real = F.normalize(u_real, dim=-1, eps=1e-8)
        u_imag = torch.zeros_like(u_real)
        return torch.stack([u_real, u_imag], dim=-1)

    def effect_values(self) -> torch.Tensor:
        """Return ``[M_eff, d, 2]`` value vectors (split-real, imag=0)."""
        v_real = self._contract_cores(self.val_cores)            # [M_eff, d]
        v_imag = torch.zeros_like(v_real)
        return torch.stack([v_real, v_imag], dim=-1)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.s_logit)

    # ── Probe / select_top_k (real-only path) ──────────────────────────────

    def project_query(self, psi: torch.Tensor) -> torch.Tensor:
        """Apply per-head probe to ``psi: [B, d, 2]`` -> ``[B, H, d, 2]``.

        v0: real probe matrix; the imag channel passes through unchanged
        (still useful upstream of the SPM, which is complex-aware)."""
        pr, pi = psi[..., 0], psi[..., 1]                       # [B, d]
        Wr = self.probe_w                                       # [H, d, d]
        qr = torch.einsum("hij,bj->bhi", Wr, pr)
        qi = torch.einsum("hij,bj->bhi", Wr, pi)
        return torch.stack([qr, qi], dim=-1)

    def probe(self, psi: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Return ``[B, H, M_eff]`` scores ``sigma(s_m) * |u_m^T q|^2``.

        Real-only inner product in v0; complex channels are summed in
        quadrature (so the score is invariant to the phase of ``q``)."""
        q = self.project_query(psi)                             # [B, H, d, 2]
        u = self.effect_directions()                            # [M, d, 2]
        ur = u[..., 0]                                          # [M, d]
        qr, qi = q[..., 0], q[..., 1]                           # [B, H, d]
        ip_r = torch.einsum("md,bhd->bhm", ur, qr)
        ip_i = torch.einsum("md,bhd->bhm", ur, qi)
        sq_amp = ip_r.square() + ip_i.square()
        gates = self.gates().view(1, 1, -1)
        return (gates * sq_amp) / max(temperature, 1e-6)

    def select_top_k(
        self,
        psi: torch.Tensor,
        k: int,
        rank: int,
        temperature: float = 1.0,
        reason_heads: int = 1,
        return_scores: bool = False,
    ):
        """Drop-in equivalent of
        :meth:`EffectAlgebraBank.select_top_k`. See that method's docstring
        for the contract on shapes; this stub preserves it exactly."""
        if reason_heads != self.n_heads:
            raise ValueError(
                f"reason_heads ({reason_heads}) must equal bank n_heads "
                f"({self.n_heads})"
            )

        scores = self.probe(psi, temperature=temperature)        # [B, H, M]
        topk_vals, topk_idx = scores.topk(k, dim=-1, largest=True)

        u_all = self.effect_directions()                         # [M, d, 2]
        w_all = self.effect_values()                             # [M, d, 2]

        idx_for_gather = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, u_all.shape[-2], 2,
        )
        u_sel = u_all.unsqueeze(0).unsqueeze(0).expand(
            scores.shape[0], scores.shape[1], -1, -1, -1,
        ).gather(2, idx_for_gather)
        w_sel = w_all.unsqueeze(0).unsqueeze(0).expand(
            scores.shape[0], scores.shape[1], -1, -1, -1,
        ).gather(2, idx_for_gather)

        weights = topk_vals.clamp_min(0.0).sqrt().unsqueeze(-1).unsqueeze(-1)
        u_sel_w = u_sel * weights
        w_sel_w = w_sel * weights

        u_for_qr = u_sel_w.transpose(-3, -2).contiguous()
        w_for_qr = w_sel_w.transpose(-3, -2).contiguous()

        # Reuse the QR routine from EffectAlgebraBank to keep behavior identical.
        from .effect_bank import EffectAlgebraBank
        U, V = EffectAlgebraBank._qr_basis(u_for_qr, w_for_qr, rank)

        if return_scores:
            return U, V, scores
        return U, V

    # ── Bookkeeping ────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        # Param accounting: cores + gates + probe.
        core_params = 0
        for c, d_c in enumerate(self.d_per_core):
            left = 1 if c == 0 else self.chi
            right = 1 if c == self.C - 1 else self.chi
            core_params += left * self.K * d_c * right
        core_params *= 2  # dir + val
        total = core_params + self.bank_size + self.n_heads * self.dim ** 2
        return (
            f"dim={self.dim}, M_eff={self.bank_size} "
            f"(K={self.K}^C={self.C}), chi={self.chi}, "
            f"n_heads={self.n_heads}, params={total}"
        )
