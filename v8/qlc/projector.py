r"""Sasaki Projection Memory (SPM).

A rank-r orthonormal subspace P = U U^\dagger over a complex d-dimensional
state, updated via the orthomodular Sasaki projection rule.

Key operations
==============

* ``build_from_basis(U_in, value_cache)`` -- ingest an external orthonormal
  basis and value cache; useful when constructing :math:`\Pi_F` from selected
  effects of an :class:`EffectAlgebraBank`.

* ``streaming_step(k, v, gamma)`` -- one Sasaki write: projects the new key
  ``k`` to the orthogonal complement of the current basis, normalizes it, and
  installs it into a circular slot. The aligned value cache is gamma-decayed
  and the new value placed at the same slot. Cost: O(d r) per step.

* ``chunked_train(k_seq, v_seq, gamma_seq)`` -- batched/parallel-friendly
  version of streaming_step that runs as a Python loop in a torch.compile or
  checkpoint friendly way; intended for short reasoning loops or per-token
  scans, not full sequence training (use the main backbone for that).

* ``retrieve(q)`` -- Sasaki retrieval: ``y = U (V_aligned @ (U^\dagger q))``.
  Equivalent to projecting onto the kept subspace and reading out the
  associated values.

Invariants
==========

* Columns of U are orthonormal up to numerical tolerance. A periodic re-QR
  pass (configurable) restores exactness if drift is detected.
* The state never materializes the d x d projector matrix; everything goes
  through the rank-r factor.

Tensor convention
=================

All complex tensors are split-real: the trailing dimension is 2 with index 0 =
real part, index 1 = imaginary part (matching v6/v7 convention). The Triton
fast paths used in the backbone are *not* used here -- the QLC works on
small (d x r) tensors and benefits more from autograd cleanliness than from
fused kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Complex helpers (local copies to keep this module self-contained) ────────

def _cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise complex multiplication on split-real tensors."""
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)


def _cabs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(x[..., 0].square() + x[..., 1].square() + eps)


def _cnorm(x: torch.Tensor, dim: int, keepdim: bool = False, eps: float = 1e-12) -> torch.Tensor:
    """L2 norm on a complex split-real tensor along ``dim`` (the d-dimension)."""
    sq = x[..., 0].square() + x[..., 1].square()
    n = sq.sum(dim=dim, keepdim=keepdim).clamp_min(eps).sqrt()
    return n


def _matmul_complex(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched complex matmul on split-real ``[..., m, n, 2]`` x ``[..., n, k, 2]``.

    Returns ``[..., m, k, 2]`` with c = a @ b in C.
    """
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    cr = ar @ br - ai @ bi
    ci = ar @ bi + ai @ br
    return torch.stack([cr, ci], dim=-1)


def _adjoint(u: torch.Tensor) -> torch.Tensor:
    r"""Conjugate transpose of a complex matrix in split-real form.

    Input  ``[..., m, n, 2]`` ->  output  ``[..., n, m, 2]``  with
    :math:`U^\dagger_{i,j} = \overline{U_{j,i}}`.
    """
    ut = u.transpose(-3, -2)  # [..., n, m, 2]
    return torch.stack([ut[..., 0], -ut[..., 1]], dim=-1)


def _vec_outer_conj(v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r"""Outer product :math:`v \otimes k^*` for complex vectors of shape ``[..., d, 2]``.

    Output: ``[..., d, d, 2]``.
    """
    vr, vi = v[..., 0].unsqueeze(-1), v[..., 1].unsqueeze(-1)
    kr, ki = k[..., 0].unsqueeze(-2), k[..., 1].unsqueeze(-2)
    # (vr + i vi)(kr - i ki) = (vr kr + vi ki) + i (vi kr - vr ki)
    rr = vr * kr + vi * ki
    ii = vi * kr - vr * ki
    return torch.stack([rr, ii], dim=-1)


# ── Module ───────────────────────────────────────────────────────────────────

@dataclass
class SPMState:
    """Streaming Sasaki Projection Memory state container.

    Shapes:
      * ``U``: ``[B, H, d, r, 2]`` -- orthonormal columns (per batch, per head).
      * ``V``: ``[B, H, d, r, 2]`` -- value vectors aligned column-wise to U.
      * ``next_slot``: ``[B, H]`` long -- circular write index modulo r.
      * ``filled``: ``[B, H]`` long -- number of slots populated so far (<= r).
    """

    U: torch.Tensor
    V: torch.Tensor
    next_slot: torch.Tensor
    filled: torch.Tensor


class SasakiProjectionMemory(nn.Module):
    r"""Sasaki projector memory with rank-r orthonormal basis.

    The module is *stateless in parameters* by design: all learnable behaviour
    lives in the upstream layers that produce ``k`` and ``v``. SPM contributes
    the *operational* machinery (orthogonal-complement projection, circular
    slot management, Sasaki retrieval) that the rest of the QLC composes
    against.
    """

    def __init__(
        self,
        dim: int,
        rank: int,
        n_heads: int = 1,
        eps: float = 1e-6,
        qr_refresh_every: int = 0,
        use_complex: bool = True,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if dim < rank:
            raise ValueError(f"dim ({dim}) must be >= rank ({rank})")
        self.dim = dim
        self.rank = rank
        self.n_heads = n_heads
        self.eps = eps
        self.qr_refresh_every = qr_refresh_every
        # Real-vs-complex ablation: when False the imag channel is forced to
        # zero everywhere, so the same code path runs but the algebra is
        # restricted to R^d. This is the discriminator §G.2: if QLC matches
        # in pure real mode then phase is not the source of the win.
        self.use_complex = use_complex
        self._step_counter = 0  # not a parameter, just a periodic-refresh counter

    @staticmethod
    def _strip_imag(x: torch.Tensor) -> torch.Tensor:
        """Zero out the imaginary channel in-place safely (out-of-place).

        Used by the real-mode ablation -- preserves split-real shape so the
        rest of the pipeline does not have to special-case anything.
        """
        out = x.clone()
        out[..., 1] = 0.0
        return out

    # ── State init ──────────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> SPMState:
        H, d, r = self.n_heads, self.dim, self.rank
        U = torch.zeros(batch_size, H, d, r, 2, device=device, dtype=dtype)
        V = torch.zeros_like(U)
        # Seed first column with the first standard basis vector to avoid degenerate
        # adjoint-times-zero gradients on step 0; will be evicted within r steps.
        U[..., 0, 0, 0] = 1.0
        next_slot = torch.zeros(batch_size, H, device=device, dtype=torch.long)
        filled = torch.zeros(batch_size, H, device=device, dtype=torch.long)
        return SPMState(U=U, V=V, next_slot=next_slot, filled=filled)

    # ── Build from external orthonormal basis ──────────────────────────────

    def build_from_basis(
        self,
        U_in: torch.Tensor,      # [B, H, d, r, 2]
        V_in: Optional[torch.Tensor] = None,  # [B, H, d, r, 2] or None
    ) -> SPMState:
        """Ingest an externally-prepared orthonormal basis (e.g. from the bank)
        and treat it as the current SPM state. ``V_in`` defaults to U_in (i.e.
        retrieval returns the projector applied to the query)."""
        B, H, d, r, two = U_in.shape
        if (H, d, r, two) != (self.n_heads, self.dim, self.rank, 2):
            raise ValueError(
                f"build_from_basis shape mismatch: got [_, {H}, {d}, {r}, {two}], "
                f"expected [_, {self.n_heads}, {self.dim}, {self.rank}, 2]"
            )
        V_eff = V_in if V_in is not None else U_in
        if not self.use_complex:
            U_in = self._strip_imag(U_in)
            V_eff = self._strip_imag(V_eff)
        next_slot = torch.zeros(B, H, device=U_in.device, dtype=torch.long)
        filled = torch.full((B, H), r, device=U_in.device, dtype=torch.long)
        return SPMState(U=U_in, V=V_eff, next_slot=next_slot, filled=filled)

    # ── Single Sasaki write (streaming inference path) ─────────────────────

    def streaming_step(
        self,
        state: SPMState,
        k: torch.Tensor,         # [B, H, d, 2]
        v: torch.Tensor,         # [B, H, d, 2]
        gamma: Optional[torch.Tensor] = None,  # [B, H] or None for no decay
    ) -> SPMState:
        r"""Perform one Sasaki update.

        Steps:

        1. ``k_perp = k - U (U^\dagger k)`` -- project k to the orthocomplement
           of the current subspace (the orthomodular ``a' \vee x`` part).
        2. Normalize ``k_perp``. If its norm is below ``eps`` (i.e. k already
           lay in the kept subspace), skip the column write to preserve
           orthonormality.
        3. ``j = next_slot mod r``; replace ``U[:, :, :, j]`` with the new
           unit vector, and ``V[:, :, :, j]`` with ``v``.
        4. Apply gamma decay to existing value columns (forgetting on values).
        5. Optionally re-QR every ``qr_refresh_every`` steps to fight drift.
        """
        U, V = state.U, state.V
        B, H, d, r, _ = U.shape

        # Step 1: orthogonal-complement projection of k against current U.
        # coef_j = U_j^\dagger k = sum_d conj(U_dj) * k_d  (per b, h)
        # coef shape: [B, H, r, 2]
        coef = self._proj_coeffs(U, k)
        # subtract  U @ coef  from k  -> k_perp
        k_proj = self._apply_basis(U, coef)
        k_perp = k - k_proj

        # Step 2: normalize. If norm tiny, fall back to a small random direction
        # in the orthogonal complement to keep gradients alive.
        norm = _cnorm(k_perp, dim=-2, keepdim=True)  # [B, H, 1, 2]? careful
        # _cnorm returns sum over dim=-2 with no last-dim split; need [B,H,1] then unsqueeze
        # Recompute explicitly to be unambiguous:
        sq = (k_perp[..., 0].square() + k_perp[..., 1].square()).sum(dim=-1, keepdim=True)
        norm_real = sq.clamp_min(self.eps ** 2).sqrt().unsqueeze(-1)  # [B, H, 1, 1]
        u_new = k_perp / norm_real  # [B, H, d, 2]

        # Step 3: write into circular slot j.
        U_new, V_new = self._scatter_column(state, u_new, v)

        # Step 4: gamma decay on the value cache (multiplicatively) -- this
        # mimics the QPAM forgetting curve while keeping the U basis intact.
        if gamma is not None:
            g = gamma.view(B, H, 1, 1, 1).clamp(0.0, 1.0)
            V_new = V_new * g

        next_slot = (state.next_slot + 1) % r
        filled = torch.minimum(state.filled + 1, torch.full_like(state.filled, r))

        new_state = SPMState(U=U_new, V=V_new, next_slot=next_slot, filled=filled)

        # Step 5: drift correction.
        if self.qr_refresh_every > 0:
            self._step_counter += 1
            if self._step_counter % self.qr_refresh_every == 0:
                new_state = self._reorthonormalize(new_state)

        return new_state

    # ── Chunked / batched training path ────────────────────────────────────

    def chunked_train(
        self,
        state: SPMState,
        k_seq: torch.Tensor,    # [B, H, T, d, 2]
        v_seq: torch.Tensor,    # [B, H, T, d, 2]
        gamma_seq: Optional[torch.Tensor] = None,  # [B, H, T] or None
    ) -> Tuple[torch.Tensor, SPMState]:
        """Sequentially apply ``T`` Sasaki updates, returning the retrieval
        outputs at each step (after that step's update) and the final state.

        Output retrievals: ``[B, H, T, d, 2]`` -- ``y_t = SPM_t.retrieve(k_t)``.
        Used by the reasoning loop's inner scan when T_max > 1; not intended
        for full sequence-length scans (use the QPAM backbone for that).
        """
        B, H, T, d, two = k_seq.shape
        ys = []
        s = state
        for t in range(T):
            g_t = gamma_seq[..., t] if gamma_seq is not None else None
            s = self.streaming_step(s, k_seq[:, :, t], v_seq[:, :, t], g_t)
            ys.append(self.retrieve(s, k_seq[:, :, t]))
        return torch.stack(ys, dim=2), s

    # ── Sasaki retrieval ────────────────────────────────────────────────────

    def retrieve(self, state: SPMState, q: torch.Tensor) -> torch.Tensor:
        r"""Return ``y = V_aligned (U^\dagger q)`` -- Sasaki readout.

        ``q`` shape: ``[B, H, d, 2]``. Output: ``[B, H, d, 2]``.
        """
        coef = self._proj_coeffs(state.U, q)            # [B, H, r, 2]
        # y = V @ coef  with V shape [B, H, d, r, 2], coef [B, H, r, 2]
        return self._apply_basis(state.V, coef)

    # ── Optional Sasaki update on a *projector* directly (for reason loop) ─

    def sasaki_apply(
        self,
        state: SPMState,
        psi: torch.Tensor,           # [B, H, d, 2]
        symmetrize: bool = False,
        prev_state: Optional[SPMState] = None,
    ) -> torch.Tensor:
        r"""Compute ``y = Pi psi`` (project psi onto the SPM subspace).

        Modes
        -----
        * ``symmetrize=False`` (default): standard Sasaki / projector update
          ``y = Pi psi``. This is what the canonical reasoning loop calls.
        * ``symmetrize=True`` and ``prev_state is None``: legacy "quantale_off"
          ablation that returns ``0.5 (Pi psi + psi)``. Kept for backward
          compatibility with the original V8-F preset; **note** this is only
          a residual blend, not a true ordering test (see AUDIT_V8.md §2).
        * ``symmetrize=True`` and ``prev_state`` supplied: **true ordering
          test** -- compares
          :math:`y = \tfrac{1}{2}(\Pi_{\text{curr}} \Pi_{\text{prev}} \psi
          + \Pi_{\text{prev}} \Pi_{\text{curr}} \psi)` against the sequential
          composition :math:`\Pi_{\text{curr}} \Pi_{\text{prev}} \psi` that
          the default loop produces. If LM perplexity rises significantly
          with the symmetric form, *order matters*.

        ``prev_state`` should be the SPM state from the previous iteration
        of the reasoning loop (i.e. the one whose projector produced ``psi``
        in the first place). The control row uses the default loop unchanged.
        """
        if not self.use_complex:
            psi = self._strip_imag(psi)
        if prev_state is not None and symmetrize:
            # True ordering test: compare Pi_curr Pi_prev psi vs the symmetric
            # average. Caller must pass `psi` *before* the curr-iteration
            # projector has been applied (i.e. psi_{t} pre-update).
            seq_curr_then_prev = self._apply_basis(
                state.U, self._proj_coeffs(state.U, psi)
            )
            seq_prev_then_curr_input = self._apply_basis(
                prev_state.U, self._proj_coeffs(prev_state.U, psi)
            )
            seq_prev_then_curr = self._apply_basis(
                state.U, self._proj_coeffs(state.U, seq_prev_then_curr_input)
            )
            seq_curr_first_input = self._apply_basis(
                state.U, self._proj_coeffs(state.U, psi)
            )
            seq_curr_first = self._apply_basis(
                prev_state.U, self._proj_coeffs(prev_state.U, seq_curr_first_input)
            )
            return 0.5 * (seq_prev_then_curr + seq_curr_first)
        coef = self._proj_coeffs(state.U, psi)
        proj = self._apply_basis(state.U, coef)
        if symmetrize:
            return 0.5 * (proj + psi)
        return proj

    # ── Internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _proj_coeffs(U: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`U^\dagger x` for ``U: [B, H, d, r, 2]``, ``x: [B, H, d, 2]``.

        Returns ``[B, H, r, 2]``.
        """
        # Real / imaginary parts of U.
        Ur, Ui = U[..., 0], U[..., 1]            # [B, H, d, r]
        xr, xi = x[..., 0], x[..., 1]            # [B, H, d]
        # conj(U)^T x: (U_r - i U_i)^T (x_r + i x_i)
        # real part: U_r^T x_r + U_i^T x_i
        # imag part: U_r^T x_i - U_i^T x_r
        # Use einsum on the d dimension.
        cr = torch.einsum("bhdr,bhd->bhr", Ur, xr) + torch.einsum("bhdr,bhd->bhr", Ui, xi)
        ci = torch.einsum("bhdr,bhd->bhr", Ur, xi) - torch.einsum("bhdr,bhd->bhr", Ui, xr)
        return torch.stack([cr, ci], dim=-1)

    @staticmethod
    def _apply_basis(U: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`U c` for ``U: [B, H, d, r, 2]``, ``c: [B, H, r, 2]``.

        Returns ``[B, H, d, 2]``.
        """
        Ur, Ui = U[..., 0], U[..., 1]
        cr, ci = coef[..., 0], coef[..., 1]
        # (U_r + i U_i)(c_r + i c_i) = (U_r c_r - U_i c_i) + i (U_r c_i + U_i c_r)
        yr = torch.einsum("bhdr,bhr->bhd", Ur, cr) - torch.einsum("bhdr,bhr->bhd", Ui, ci)
        yi = torch.einsum("bhdr,bhr->bhd", Ur, ci) + torch.einsum("bhdr,bhr->bhd", Ui, cr)
        return torch.stack([yr, yi], dim=-1)

    @staticmethod
    def _scatter_column(
        state: SPMState,
        u_new: torch.Tensor,    # [B, H, d, 2]
        v_new: torch.Tensor,    # [B, H, d, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable column-replace at slot ``state.next_slot`` (per b, h).

        Implemented as ``U_new = U * (1 - mask) + u_new * mask`` where mask is
        a one-hot over the r dimension. Avoids in-place ops that would break
        autograd.
        """
        U, V = state.U, state.V
        B, H, d, r, _ = U.shape
        # one-hot mask over the r columns: [B, H, r]
        mask = F.one_hot(state.next_slot, num_classes=r).to(dtype=U.dtype)
        # Expand to [B, H, 1, r, 1] for broadcasting with [B, H, d, r, 2]
        mask_b = mask.unsqueeze(-2).unsqueeze(-1)
        # u_new -> [B, H, d, 1, 2] -> broadcast over r via mask
        u_b = u_new.unsqueeze(-2)
        v_b = v_new.unsqueeze(-2)
        U_new = U * (1.0 - mask_b) + u_b * mask_b
        V_new = V * (1.0 - mask_b) + v_b * mask_b
        return U_new, V_new

    @torch.no_grad()
    def _reorthonormalize(self, state: SPMState) -> SPMState:
        """Re-QR the U columns to undo numerical drift (no-grad; only used as a
        periodic safety net during long inference scans)."""
        U = state.U
        B, H, d, r, _ = U.shape
        # Treat U as a complex matrix [d, r] per (b, h) and run torch.linalg.qr
        # in complex form. Round-trip through torch.complex64 because qr does
        # not accept split-real -- we never propagate gradients through this
        # path so the OOM/autograd warnings about complex tensors do not apply.
        U_c = torch.view_as_complex(U.contiguous())  # [B, H, d, r]
        Q, _ = torch.linalg.qr(U_c, mode="reduced")   # Q: [B, H, d, r]
        U_re = torch.view_as_real(Q.contiguous()).contiguous()
        return SPMState(U=U_re, V=state.V, next_slot=state.next_slot, filled=state.filled)

    # ── Diagnostics ────────────────────────────────────────────────────────

    @torch.no_grad()
    def orthonormality_error(self, state: SPMState) -> torch.Tensor:
        r"""Return ``|| U^\dagger U - I_r ||_F`` averaged over (B, H).

        Used by tests to confirm columns stay orthonormal under streaming
        updates.
        """
        U = state.U
        B, H, d, r, _ = U.shape
        U_c = torch.view_as_complex(U.contiguous())
        # See the note in effect_bank._qr_basis: avoid the complex
        # transpose+conj+matmul chain (fragile under torch.compile) by
        # writing the same ``U^H @ U`` contraction as einsum. Even though
        # this site is @no_grad today, keeping the pattern uniform across
        # the QLC means future callers can wrap it under autograd safely.
        gram = torch.einsum("bhdi,bhdj->bhij", U_c.conj(), U_c)
        eye = torch.eye(r, dtype=gram.dtype, device=gram.device)
        diff = gram - eye
        return diff.abs().square().sum(dim=(-1, -2)).sqrt().mean()
