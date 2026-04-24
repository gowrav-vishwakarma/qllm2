r"""Effect-Algebra Bank.

A bank of ``M`` learnable rank-1 Hermitian *effects*

.. math::

   E_m = \sigma(s_m) \cdot u_m u_m^\dagger,
   \qquad u_m \in \mathbb{C}^d, \; \|u_m\| = 1.

Each effect satisfies :math:`0 \le E_m \le I` automatically because
:math:`\sigma(s_m) \in [0, 1]`. This makes the bank an honest
*effect algebra* (Foulis-Bennett 1994; see plan ref to Coecke notes), where
the partial sum

.. math:: e \oplus e' = e + e' \quad \text{when} \quad e + e' \le I

is well defined for selected disjoint effects and saturates at the identity
for the *whole* bank only if the rows are orthogonal -- a structure we
do *not* enforce, since the bank is parameter memory and we *want*
overlapping rows to model correlated facts.

The bank exposes:

* ``probe(psi)`` -- for query state :math:`\psi`, return scores
  :math:`\langle\psi | E_m | \psi\rangle = \sigma(s_m) \cdot |u_m^\dagger \psi|^2`.
  Used by the QLC reasoning loop to pick the top-k facts each iteration.

* ``select_top_k(psi, k)`` -- return the (orthonormalized) basis ``U``
  spanning the top-k selected ``u_m`` vectors plus their value vectors,
  ready to feed :class:`SasakiProjectionMemory.build_from_basis`.

* ``infonce_loss(psi_pos, psi_neg)`` -- contrastive auxiliary used in
  Stage B to push the bank toward routing entity-mask cloze positives
  to the same effects as their unmasked context.

Every effect carries an associated *value vector* ``w_m`` so that retrieval
through SPM has a non-trivial readout (otherwise SPM with V_aligned = U is a
plain projector that returns ``Pi psi``; with V_aligned = W it returns the
fact's stored content).

Tensor convention is the same split-real ``[..., d, 2]`` as the rest of v8.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cnormalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize a complex split-real tensor ``[..., d, 2]`` along the d dim.

    Treats the trailing 2 as the real/imag split (not a feature axis to sum
    over) and the immediately-preceding axis as the d-vector axis. The norm
    is :math:`\\sqrt{\\sum_d (x_r^2 + x_i^2)}`.
    """
    sq = (x[..., 0].square() + x[..., 1].square()).sum(dim=-1, keepdim=True)
    norm = sq.clamp_min(eps).sqrt().unsqueeze(-1)
    return x / norm


# ── Module ───────────────────────────────────────────────────────────────────

class EffectAlgebraBank(nn.Module):
    r"""Bank of :math:`M` learnable rank-1 effects with associated value vectors.

    Parameters
    ----------
    dim
        Complex dimension :math:`d` of effect vectors.
    bank_size
        Number of effects :math:`M`.
    n_heads
        Number of reasoning heads sharing the bank's underlying embedding
        space; each head gets its own probe head but the effects are shared.
        Default 1 (single-head reasoning).
    init_scale
        Std of the Gaussian used to initialize the raw ``u_m`` vectors before
        normalization.
    """

    def __init__(
        self,
        dim: int,
        bank_size: int,
        n_heads: int = 1,
        init_scale: float = 0.02,
        gate_init: float = -2.0,
    ):
        super().__init__()
        self.dim = dim
        self.bank_size = bank_size
        self.n_heads = n_heads

        # Effect direction vectors (raw; normalized in forward).
        # Shape: [M, d, 2]  (split-real).
        self.u_real = nn.Parameter(torch.randn(bank_size, dim) * init_scale)
        self.u_imag = nn.Parameter(torch.randn(bank_size, dim) * init_scale)

        # Per-effect value vector (can drift from u_m). [M, d, 2].
        self.w_real = nn.Parameter(torch.randn(bank_size, dim) * init_scale)
        self.w_imag = nn.Parameter(torch.randn(bank_size, dim) * init_scale)

        # Effect gate logits (sigmoid -> [0,1] amplitude). Init negative so the
        # bank starts mostly silent and the LM signal must "wake up" useful
        # effects. [M].
        self.s_logit = nn.Parameter(torch.full((bank_size,), gate_init))

        # Per-head probe projection: each reasoning head can ask the bank a
        # different question by transforming psi before scoring against u_m.
        # Implemented as a per-head complex linear via two real matrices each.
        # When n_heads = 1, this is effectively a single head Q -> Q transform.
        self.probe_real = nn.Parameter(torch.empty(n_heads, dim, dim))
        self.probe_imag = nn.Parameter(torch.empty(n_heads, dim, dim))
        for h in range(n_heads):
            nn.init.orthogonal_(self.probe_real[h], gain=(1.0 / math.sqrt(dim)))
            nn.init.orthogonal_(self.probe_imag[h], gain=(1.0 / math.sqrt(dim)))

    # ── Effect / value tensor accessors (normalized) ───────────────────────

    def effect_directions(self) -> torch.Tensor:
        """Return the M unit-norm effect vectors as ``[M, d, 2]``."""
        u = torch.stack([self.u_real, self.u_imag], dim=-1)
        return _cnormalize(u)

    def effect_values(self) -> torch.Tensor:
        """Return the M raw value vectors as ``[M, d, 2]`` (no normalization)."""
        return torch.stack([self.w_real, self.w_imag], dim=-1)

    def gates(self) -> torch.Tensor:
        """Return per-effect gate ``sigma(s_m)`` in ``[0, 1]`` -- shape ``[M]``."""
        return torch.sigmoid(self.s_logit)

    # ── Probe heads ────────────────────────────────────────────────────────

    def project_query(self, psi: torch.Tensor) -> torch.Tensor:
        r"""Apply the per-head probe transform to ``psi``.

        Input  ``psi: [B, d, 2]`` -> output ``[B, H, d, 2]``.
        For each head h: q_h = (probe_real[h] + i probe_imag[h]) psi.
        """
        # psi: [B, d, 2] -> [B, 1, d] for real/imag
        pr, pi = psi[..., 0], psi[..., 1]            # [B, d]
        Wr = self.probe_real                          # [H, d, d]
        Wi = self.probe_imag                          # [H, d, d]
        # (Wr + iWi)(pr + ipi) = (Wr pr - Wi pi) + i (Wr pi + Wi pr)
        qr = torch.einsum("hij,bj->bhi", Wr, pr) - torch.einsum("hij,bj->bhi", Wi, pi)
        qi = torch.einsum("hij,bj->bhi", Wr, pi) + torch.einsum("hij,bj->bhi", Wi, pr)
        return torch.stack([qr, qi], dim=-1)

    # ── Probe scoring ──────────────────────────────────────────────────────

    def probe(
        self,
        psi: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        r"""Compute :math:`\text{score}_{b,h,m} = \sigma(s_m) |u_m^\dagger q_{b,h}|^2 / \tau`.

        Returns ``[B, H, M]`` of raw effect scores (pre-softmax). Higher score
        = effect more likely to be selected.
        """
        q = self.project_query(psi)                                  # [B, H, d, 2]
        u = self.effect_directions()                                 # [M, d, 2]
        # Inner product u_m^H q for every (b, h, m).
        ur, ui = u[..., 0], u[..., 1]                                # [M, d]
        qr, qi = q[..., 0], q[..., 1]                                # [B, H, d]
        # (u_r - i u_i)^T (q_r + i q_i) = (u_r^T q_r + u_i^T q_i) + i(u_r^T q_i - u_i^T q_r)
        ip_r = torch.einsum("md,bhd->bhm", ur, qr) + torch.einsum("md,bhd->bhm", ui, qi)
        ip_i = torch.einsum("md,bhd->bhm", ur, qi) - torch.einsum("md,bhd->bhm", ui, qr)
        sq_amp = ip_r.square() + ip_i.square()                       # [B, H, M]
        gates = self.gates().view(1, 1, -1)                          # [1, 1, M]
        return (gates * sq_amp) / max(temperature, 1e-6)

    # ── Top-k selection -> orthonormal basis for SPM ───────────────────────
    def select_top_k(
        self,
        psi: torch.Tensor,
        k: int,
        rank: int,
        temperature: float = 1.0,
        reason_heads: int = 1,
        return_scores: bool = False,
        return_topk_idx: bool = False,
    ):
        r"""Select top-k effects per (batch, head) and build a rank-r orthonormal
        basis ``U`` plus aligned values ``V`` ready for
        :meth:`SasakiProjectionMemory.build_from_basis`.

        Strategy
        --------
        1. Score all M effects with :meth:`probe`.
        2. Take top-k indices per (b, h).
        3. Gather the corresponding ``u_m`` (effect direction) and ``w_m``
           (value) vectors. Weight them by ``sqrt(gate * |inner|^2)`` so the
           subsequent QR keeps the strongest effects' directions.
        4. Run a complex QR on the (d x k) matrix to get an orthonormal basis
           of size r = min(k, rank). If r < rank, pad with zero columns.

        Returns
        -------
        U : ``[B, H_reason, d, rank, 2]``
        V : ``[B, H_reason, d, rank, 2]``  (values aligned column-wise)
        scores : optional, ``[B, H_reason, M]`` (returned if return_scores=True)
        topk_idx : optional, ``[B, H_reason, k]`` long (returned if
                   return_topk_idx=True). Used by the per-iter bank-overlap
                   diagnostic in :class:`QuantumLogicCore` so we can compute
                   Jaccard overlap of the selected effects across iterations.
                   No autograd cost: indices are int64.
        """
        if reason_heads != self.n_heads:
            # Allow caller to subset or broadcast heads if it really wants,
            # but the common case is reason_heads == n_heads.
            raise ValueError(
                f"reason_heads ({reason_heads}) must equal bank n_heads "
                f"({self.n_heads})"
            )

        scores = self.probe(psi, temperature=temperature)            # [B, H, M]
        topk_vals, topk_idx = scores.topk(k, dim=-1, largest=True)   # [B, H, k]

        u_all = self.effect_directions()                              # [M, d, 2]
        w_all = self.effect_values()                                  # [M, d, 2]

        # Gather selected vectors per (b, h, k). topk_idx: [B, H, k].
        #
        # IMPORTANT: the natural-looking expression
        #
        #   u_sel = (
        #       u_all.unsqueeze(0).unsqueeze(0)            # [1, 1, M, d, 2]
        #       .expand(B, H, -1, -1, -1)                  # [B, H, M, d, 2] view
        #       .gather(2, idx_for_gather)                 # [B, H, k, d, 2]
        #   )
        #
        # has a backward that materializes the full ``[B, H, M, d, 2]`` dense
        # gradient buffer before scatter-summing back to the bank's
        # ``[M, d, 2]``. For the medium preset (B*T=6144, M=2048, d=384)
        # this is 6144 * 2048 * 384 * 2 * 4 = 38.6 GiB and OOMs the card.
        # The same pathology exists in eager mode -- it's not inductor's
        # fault, it's the backward formula for ``gather`` over an expanded
        # view. The smoke preset's small M*d kept it under the threshold.
        #
        # ``index_select`` does the same forward gather but its backward
        # only allocates a single dense gradient of size ``[M, d, 2]``
        # (~6 MB), populated via the standard sparse scatter_add_. We
        # flatten (B, H, k) into one index dimension, index_select once,
        # then reshape back to [B, H, k, d, 2].
        flat_idx = topk_idx.reshape(-1)                               # [B*H*k]
        out_shape = (*topk_idx.shape, u_all.shape[1], u_all.shape[2])  # [B, H, k, d, 2]
        u_sel = u_all.index_select(0, flat_idx).reshape(out_shape)
        w_sel = w_all.index_select(0, flat_idx).reshape(out_shape)

        # Weight selected directions by sqrt(score) so QR keeps the dominant
        # facts' span; this also lets gradients flow through topk_vals.
        weights = topk_vals.clamp_min(0.0).sqrt().unsqueeze(-1).unsqueeze(-1)  # [B,H,k,1,1]
        u_sel_w = u_sel * weights
        w_sel_w = w_sel * weights

        # Reorder dims to put (d, k) at the end for QR: [B, H, k, d, 2] -> [B, H, d, k, 2]
        u_for_qr = u_sel_w.transpose(-3, -2).contiguous()
        w_for_qr = w_sel_w.transpose(-3, -2).contiguous()
        U, V = self._qr_basis(u_for_qr, w_for_qr, rank)

        if return_scores and return_topk_idx:
            return U, V, scores, topk_idx
        if return_scores:
            return U, V, scores
        if return_topk_idx:
            return U, V, topk_idx
        return U, V

    # NOTE: ``torch.compile`` (inductor backend) cannot reliably trace the
    # backward of ``torch.linalg.qr`` on complex tensors. The autograd formula
    # for QR on complex internally chains ``transpose(-1,-2)`` and ``conj()``
    # operations, both of which feed inductor's complex ``add`` decomposition
    # that asserts ``stride(-1) == 1`` when viewing ComplexFloat as Float.
    # The transpose/conj produces a non-contiguous complex gradient and the
    # decomp aborts (RuntimeError ``stride(-1) must be 1 ...``).
    #
    # Forcing this small slice of compute (~one call per ACT iteration, ~3-5
    # per forward) to run in eager mode adds a single graph break per call
    # but lets inductor compile the rest of the model normally (backbone,
    # sasaki_apply, halt heads -- the ~99% of compute). Net cost is a few
    # hundred microseconds per forward, well below other overheads.
    @staticmethod
    @torch._dynamo.disable
    def _qr_basis(
        u_mat: torch.Tensor,    # [B, H, d, k, 2]  weighted selected directions
        w_mat: torch.Tensor,    # [B, H, d, k, 2]  weighted associated values
        rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a complex QR on ``u_mat`` and project ``w_mat`` accordingly.

        We use ``torch.linalg.qr`` in complex form (not split-real) since QR
        gradients are well supported on complex tensors in modern PyTorch
        (in eager). The ``@torch._dynamo.disable`` above forces this function
        to run in eager mode under ``torch.compile`` -- see the long note
        right above this method for the rationale.

        Padding with zeros if k < rank, truncation if k > rank.
        """
        B, H, d, k, _ = u_mat.shape
        u_c = torch.view_as_complex(u_mat.contiguous())  # [B, H, d, k]
        # Reduced QR: Q has shape [B, H, d, min(d, k)].
        Q, R = torch.linalg.qr(u_c, mode="reduced")
        cols = Q.shape[-1]

        # Pad / truncate to exactly `rank` columns.
        if cols < rank:
            pad = torch.zeros(B, H, d, rank - cols, dtype=Q.dtype, device=Q.device)
            Q = torch.cat([Q, pad], dim=-1)
        elif cols > rank:
            Q = Q[..., :rank]

        # Map values into the same orthonormal frame: V = w_mat @ R^{-1} truncated
        # to first `rank` columns. To avoid solving a linear system (R may be
        # ill-conditioned when scores collapse), we instead compute V as the
        # projection of the stacked value matrix onto the new basis:
        #     V = Q @ (Q^H @ w_mat).
        # This guarantees V lies in span(Q) and aligns column-wise.
        w_c = torch.view_as_complex(w_mat.contiguous())     # [B, H, d, k]
        # Project w onto the QR basis -> coefficients of size [B, H, rank, k].
        # Mathematically this is ``Q^H @ w_c``. The naive expression
        # ``Q.transpose(-1, -2).conj() @ w_c`` is fragile under
        # torch.compile/inductor: the ``transpose`` puts the last dim at
        # stride != 1 and its backward (``TransposeBackward0``) forces the
        # gradient through inductor's complex ``add`` decomposition, which
        # tries to ``view ComplexFloat as Float`` and asserts
        # ``stride(-1) == 1``. ``einsum`` expresses the same contraction
        # without any complex-tensor transpose, so neither forward nor
        # backward produces the bad stride pattern.
        coeff = torch.einsum(
            "bhdr,bhdk->bhrk", Q.conj(), w_c
        )                                                   # [B, H, rank, k]
        # Average coeffs across the k input columns (each Q column gets a
        # weighted readout). This keeps V and U column-aligned by construction.
        # Use a diagonal pick: V_i = Q[:, :, :, i] * coeff[:, :, i, i].
        # If k < rank we only have valid coefficients up to k.
        eff_cols = min(rank, k)
        diag_coeff = coeff[..., :eff_cols, :eff_cols].diagonal(dim1=-2, dim2=-1)  # [B,H,eff]
        # Build V column by column: V[:, :, :, i] = Q[:, :, :, i] * diag_coeff[i]
        # Vectorize via broadcasting.
        Q_eff = Q[..., :eff_cols]                            # [B, H, d, eff]
        V_eff = Q_eff * diag_coeff.unsqueeze(-2)             # [B, H, d, eff]

        if eff_cols < rank:
            pad_v = torch.zeros(B, H, d, rank - eff_cols, dtype=V_eff.dtype, device=V_eff.device)
            V = torch.cat([V_eff, pad_v], dim=-1)
        else:
            V = V_eff

        U_re = torch.view_as_real(Q.contiguous()).contiguous()
        V_re = torch.view_as_real(V.contiguous()).contiguous()
        return U_re, V_re

    # ── InfoNCE auxiliary (Stage B) ────────────────────────────────────────

    def infonce_loss(
        self,
        psi_pos: torch.Tensor,    # [B, d, 2] -- query state aligned with positive entity
        psi_neg: torch.Tensor,    # [B, N, d, 2] -- distractor states (e.g. other entities in the batch)
        gold_effect_idx: torch.Tensor,  # [B] long -- index of the effect that should win
        temperature: float = 0.1,
    ) -> torch.Tensor:
        r"""InfoNCE on effect routing.

        Encourages :math:`\text{score}(\psi_{pos}, E_{gold}) \gg \text{score}(\psi_{neg}, E_{gold})`.
        Used by Stage B's synthetic entity-attribute QA generator to teach the
        bank to route entity-related queries to dedicated effects.

        ``psi_pos``, ``psi_neg`` are *unprojected* states (the bank's own probe
        head is applied internally). ``gold_effect_idx[b]`` is the effect that
        should win for sample b.
        """
        scores_pos = self.probe(psi_pos, temperature=temperature)         # [B, H, M]
        # Combine batch of negatives by reshape: [B*N, d, 2]
        B, N, d, _ = psi_neg.shape
        scores_neg = self.probe(psi_neg.reshape(B * N, d, 2), temperature=temperature)
        scores_neg = scores_neg.view(B, N, self.n_heads, self.bank_size)  # [B, N, H, M]

        # Pull out the gold effect's scores.
        # scores_pos: [B, H, M] -> gather gold along M -> [B, H]
        gold_idx = gold_effect_idx.view(B, 1, 1).expand(-1, self.n_heads, 1)
        pos_score = scores_pos.gather(-1, gold_idx).squeeze(-1)           # [B, H]
        neg_score = scores_neg.gather(-1, gold_idx.unsqueeze(1).expand(-1, N, -1, -1)).squeeze(-1)
        # neg_score: [B, N, H]. Concatenate pos along N axis.
        all_score = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)  # [B, 1+N, H]
        # Cross entropy: target index 0 is positive.
        # Average over heads.
        all_score = all_score.permute(0, 2, 1).reshape(B * self.n_heads, 1 + N)
        target = torch.zeros(B * self.n_heads, dtype=torch.long, device=all_score.device)
        return F.cross_entropy(all_score, target)

    # ── Bookkeeping ────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, M={self.bank_size}, n_heads={self.n_heads}, "
            f"params={(2 + 2) * self.bank_size * self.dim + self.bank_size + 2 * self.n_heads * self.dim ** 2}"
        )
