r"""OrthoHalt: orthocomplement-based halting head for the QLC reasoning loop.

For a learned target rank-1 effect :math:`\Pi_\text{target}(\psi) = u u^\dagger`
with ``u = MLP(psi)`` (unit-norm), we read three masses

.. math::
   \alpha = |u^\dagger \psi|^2, \quad
   \beta  = |\psi - u (u^\dagger \psi)|^2, \quad
   \gamma = 1 - \alpha - \beta.

In a *classical* (real, commuting, distributive) world, :math:`\gamma = 0`
identically. In our setting :math:`\gamma > 0` whenever
:math:`[\psi, \Pi_\text{target}] \ne 0` -- i.e. precisely when the
distributivity of classical logic fails. This is the operational signal that
the model "doesn't know yet" and should keep iterating, separate from the
binary halt-yes / halt-no decision encoded by :math:`\alpha` vs
:math:`\beta`.

The output is a 3-way logits tensor over (halt-yes, halt-no/refetch, continue)
plus the raw triple :math:`(\alpha, \beta, \gamma)` for diagnostics.

Pondering follows the ACT pattern (Graves 2016): each step emits a
"continue" probability, the loop runs until cumulative continue mass exceeds
a threshold (or T_max is hit), and the per-step `continue` probabilities are
summed into a ``ponder_cost`` returned alongside the final state. The
trainer adds ``ponder_lambda * ponder_cost`` to the LM loss.

The ``orthohalt_off`` ablation (V8-G) replaces this module with a plain MLP
halt head that ignores the algebraic readout entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_real(z: torch.Tensor) -> torch.Tensor:
    """Concatenate real and imag halves of a split-real tensor along the d dim."""
    return torch.cat([z[..., 0], z[..., 1]], dim=-1)


def _cnormalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    sq = (x[..., 0].square() + x[..., 1].square()).sum(dim=-1, keepdim=True)
    norm = sq.clamp_min(eps).sqrt().unsqueeze(-1)
    return x / norm


@dataclass
class HaltOutput:
    """Per-step halt readout.

    Fields
    ------
    logits : ``[B, H, 3]``  -- softmax categories (halt-yes, halt-no, continue).
    abg    : ``[B, H, 3]``  -- raw (alpha, beta, gamma) masses.
    p_halt : ``[B, H]``     -- probability of halting *now* (= alpha + beta = 1 - p_continue
                               in the classical world; here = halt-yes + halt-no).
    p_yes  : ``[B, H]``     -- probability of "halt-yes" (positive answer).
    """

    logits: torch.Tensor
    abg: torch.Tensor
    p_halt: torch.Tensor
    p_yes: torch.Tensor


# ── Modules ──────────────────────────────────────────────────────────────────

class OrthoHalt(nn.Module):
    r"""Orthocomplement halting head with ``(alpha, beta, gamma)`` readout.

    Parameters
    ----------
    dim : int
        Complex dimension of the input state.
    n_heads : int
        Number of reasoning heads (each gets its own target-effect MLP).
    hidden_mult : int
        Hidden width multiplier for the target-effect MLP (operates on real
        2*dim concatenation).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 1,
        hidden_mult: int = 1,
        unsharp_target: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.unsharp_target = unsharp_target
        hidden = max(dim, hidden_mult * dim)

        # Target-effect generator: psi -> u (unit-norm complex vector) per head.
        # We project from real(2d) -> hidden -> 2*d*H, then split into per-head
        # real/imag and normalize.
        self.target_mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * dim * n_heads),
        )

        # Per-head unsharp gate: when `unsharp_target=True`, the target effect
        # is E = sigma(g) * u u^H (so 0 <= E <= I but E^2 != E). gamma then
        # measures the gate deficit (1 - sigma(g)) * |u^H psi|^2 instead of
        # being structurally pinned to zero (see AUDIT_V8.md §1).
        self.target_gate = nn.Parameter(torch.zeros(n_heads))

        # 3-way classification head from (alpha, beta, gamma).
        # We *do* let it learn arbitrary mixing, but initialize so gamma -> "continue"
        # and (alpha, beta) -> (halt-yes, halt-no) by default.
        self.cls_head = nn.Linear(3, 3, bias=True)
        with torch.no_grad():
            self.cls_head.weight.copy_(torch.eye(3) * 4.0)
            self.cls_head.bias.zero_()

    def target_vector(self, psi: torch.Tensor) -> torch.Tensor:
        r"""Return a per-head unit-norm target vector :math:`u(\psi)`.

        Input ``psi: [B, d, 2]``, output ``[B, H, d, 2]``.
        """
        B = psi.shape[0]
        x = _to_real(psi)                                         # [B, 2d]
        u = self.target_mlp(x)                                    # [B, 2dH]
        u = u.view(B, self.n_heads, self.dim, 2)                  # [B, H, d, 2]
        return _cnormalize(u)

    def abg(self, psi_per_head: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`(\alpha, \beta, \gamma)` for ``psi_per_head: [B, H, d, 2]``,
        ``u: [B, H, d, 2]`` (unit norm).

        Returns ``[B, H, 3]``.

        With ``unsharp_target=False`` (default, legacy): the target is the
        sharp projector :math:`P = u u^H` and we get
        :math:`\alpha + \beta = \|\psi\|^2`, so :math:`\gamma` collapses to
        the renorm deficit (typically 0 in this code path). See
        :doc:`AUDIT_V8.md` §1.

        With ``unsharp_target=True``: the target is the unsharp effect
        :math:`E = \sigma(g) u u^H`, so

        .. math::
            \alpha = \sigma(g)\,|u^H\psi|^2,\;
            \gamma = (1 - \sigma(g))\,|u^H\psi|^2,\;
            \beta = \|\psi\|^2 - |u^H\psi|^2.

        Now :math:`\alpha + \beta + \gamma = \|\psi\|^2` and :math:`\gamma`
        carries the gate deficit, so it is non-zero whenever :math:`g` is
        finite and the state has any overlap with the target.
        """
        ur, ui = u[..., 0], u[..., 1]                              # [B, H, d]
        pr, pi = psi_per_head[..., 0], psi_per_head[..., 1]
        ip_r = (ur * pr).sum(dim=-1) + (ui * pi).sum(dim=-1)       # [B, H]
        ip_i = (ur * pi).sum(dim=-1) - (ui * pr).sum(dim=-1)
        amp_sq = ip_r.square() + ip_i.square()                     # [B, H] -- |u^H psi|^2

        psi_norm_sq = (pr.square() + pi.square()).sum(dim=-1)      # [B, H]
        beta = (psi_norm_sq - amp_sq).clamp_min(0.0)

        if self.unsharp_target:
            gate = torch.sigmoid(self.target_gate).view(1, self.n_heads)  # [1, H]
            alpha = gate * amp_sq
            gamma = (1.0 - gate) * amp_sq
        else:
            alpha = amp_sq
            gamma = (1.0 - alpha - beta).clamp_min(0.0)

        return torch.stack([alpha, beta, gamma], dim=-1)

    def forward(self, psi: torch.Tensor) -> HaltOutput:
        r"""Run the orthocomplement readout on ``psi: [B, d, 2]``.

        Returns a :class:`HaltOutput` with per-head halt logits and raw
        ``(alpha, beta, gamma)``.
        """
        u = self.target_vector(psi)                                # [B, H, d, 2]
        psi_h = psi.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [B, H, d, 2]
        abg = self.abg(psi_h, u)                                    # [B, H, 3]
        logits = self.cls_head(abg)                                 # [B, H, 3]
        probs = logits.softmax(dim=-1)
        p_yes = probs[..., 0]
        p_no = probs[..., 1]
        p_halt = p_yes + p_no
        return HaltOutput(logits=logits, abg=abg, p_halt=p_halt, p_yes=p_yes)


class DeltaHalt(nn.Module):
    r"""Empirical halt: halt when the iteration's update magnitude plateaus.

    Per-iteration input is the *delta* :math:`\Delta_t = \psi_t - \psi_{t-1}`
    in split-real form. We pass its squared norm through a tiny scalar gate
    that produces ``p_halt`` directly (a learned monotonic threshold on
    "how much did this step change the state"). When delta is small, the
    gate fires, the loop halts.

    No algebraic readout. No (alpha, beta, gamma) -- abg is reported as
    zeros for diagnostic compatibility. This is one of the empirical halt
    options recommended in ``v8_classical_rethink_44e4a93c.plan.md`` §G.6.

    The first iteration has no previous state; in that case the caller
    should pass ``psi_prev = psi`` (zero delta), which keeps p_halt at the
    bias value (i.e. "do not halt yet").
    """

    def __init__(self, dim: int, n_heads: int = 1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        # Per-head threshold + temperature; mapped to halt-yes/halt-no/continue.
        self.threshold = nn.Parameter(torch.zeros(n_heads))
        self.temperature = nn.Parameter(torch.ones(n_heads))
        # Re-use a 3-way head so the rest of the loop sees the same shape;
        # we route low-delta -> halt-yes, high-delta -> continue.
        self.cls_head = nn.Linear(2, 3, bias=True)
        with torch.no_grad():
            # Default mapping: tiny delta -> halt-yes, big delta -> continue,
            # halt-no kept low.
            self.cls_head.weight.copy_(torch.tensor([
                [-1.0, 0.0],   # halt-yes from -log(delta)
                [0.0, 0.0],    # halt-no neutral
                [1.0, 0.0],    # continue from +log(delta)
            ]))
            self.cls_head.bias.copy_(torch.tensor([0.0, -2.0, 0.0]))

    def forward(
        self,
        psi: torch.Tensor,
        psi_prev: Optional[torch.Tensor] = None,
    ) -> HaltOutput:
        B = psi.shape[0]
        if psi_prev is None:
            delta_sq = torch.zeros(B, self.n_heads, device=psi.device, dtype=psi.dtype)
        else:
            d = psi - psi_prev
            delta_sq = (d[..., 0].square() + d[..., 1].square()).sum(dim=-1)
            # If psi has shape [B, d, 2] (single-head input), broadcast
            # delta_sq to all heads.
            if delta_sq.dim() == 1:
                delta_sq = delta_sq.unsqueeze(-1).expand(-1, self.n_heads)

        log_delta = torch.log(delta_sq.clamp_min(1e-12)) / max(1.0, float(self.dim))
        log_delta = log_delta - self.threshold.view(1, self.n_heads)
        log_delta = log_delta * F.softplus(self.temperature.view(1, self.n_heads))
        feats = torch.stack([log_delta, torch.ones_like(log_delta)], dim=-1)  # [B, H, 2]
        logits = self.cls_head(feats)                                          # [B, H, 3]
        probs = logits.softmax(dim=-1)
        p_yes = probs[..., 0]
        p_no = probs[..., 1]
        p_halt = p_yes + p_no
        abg = torch.zeros(B, self.n_heads, 3, device=psi.device, dtype=psi.dtype)
        return HaltOutput(logits=logits, abg=abg, p_halt=p_halt, p_yes=p_yes)


class EntropyHalt(nn.Module):
    r"""Empirical halt driven by predictive entropy on a small surrogate head.

    A tiny linear ``surrogate: psi -> R^{vocab_surrogate}`` is trained jointly
    with the QLC. Per iteration we compute the surrogate distribution's
    entropy; when it stops decreasing across iterations, halt.

    This approximates the §G.6 recommendation ("halt when next-token entropy
    stops decreasing") without paying the cost of the full LM head at every
    iteration. The surrogate is small (vocab_surrogate ~256) so the extra
    parameter count is negligible.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 1,
        vocab_surrogate: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.vocab_surrogate = vocab_surrogate
        # Tiny surrogate head -- consumes the real-2d projection.
        self.surrogate = nn.Linear(2 * dim, vocab_surrogate, bias=True)
        # Scalar per-head temperature on the entropy delta -> halt prob map.
        self.temperature = nn.Parameter(torch.ones(n_heads))
        self.threshold = nn.Parameter(torch.zeros(n_heads))
        self.cls_head = nn.Linear(2, 3, bias=True)
        with torch.no_grad():
            self.cls_head.weight.copy_(torch.tensor([
                [-1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ]))
            self.cls_head.bias.copy_(torch.tensor([0.0, -2.0, 0.0]))

    def _entropy(self, psi: torch.Tensor) -> torch.Tensor:
        x = _to_real(psi)                                # [B, 2d]
        logits = self.surrogate(x)                        # [B, vocab_surrogate]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)           # [B]

    def forward(
        self,
        psi: torch.Tensor,
        psi_prev: Optional[torch.Tensor] = None,
    ) -> HaltOutput:
        B = psi.shape[0]
        H = self.n_heads
        ent_curr = self._entropy(psi)                     # [B]
        if psi_prev is None:
            ent_delta = torch.zeros(B, device=psi.device, dtype=psi.dtype)
        else:
            ent_prev = self._entropy(psi_prev)
            ent_delta = ent_prev - ent_curr               # >0 means improving (lower entropy)
        ent_delta = ent_delta.unsqueeze(-1).expand(-1, H) - self.threshold.view(1, H)
        ent_delta = ent_delta * F.softplus(self.temperature.view(1, H))
        feats = torch.stack([ent_delta, torch.ones_like(ent_delta)], dim=-1)  # [B, H, 2]
        logits = self.cls_head(feats)
        probs = logits.softmax(dim=-1)
        p_yes = probs[..., 0]
        p_no = probs[..., 1]
        p_halt = p_yes + p_no
        abg = torch.zeros(B, H, 3, device=psi.device, dtype=psi.dtype)
        return HaltOutput(logits=logits, abg=abg, p_halt=p_halt, p_yes=p_yes)


class MLPHalt(nn.Module):
    """Plain MLP halt head -- the V8-G ablation row.

    Has the same input/output signature as :class:`OrthoHalt` so the reasoning
    loop can swap them transparently. Crucially it never sees alpha/beta/gamma,
    so any signal the algebraic readout was carrying must be re-learned from
    scratch by the MLP.
    """

    def __init__(self, dim: int, n_heads: int = 1, hidden_mult: int = 1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        hidden = max(dim, hidden_mult * dim)
        self.head = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3 * n_heads),
        )

    def forward(self, psi: torch.Tensor) -> HaltOutput:
        B = psi.shape[0]
        x = _to_real(psi)
        logits = self.head(x).view(B, self.n_heads, 3)
        probs = logits.softmax(dim=-1)
        p_yes = probs[..., 0]
        p_no = probs[..., 1]
        p_halt = p_yes + p_no
        # Provide a zero (alpha, beta, gamma) for diagnostic compatibility.
        abg = torch.zeros(B, self.n_heads, 3, device=psi.device, dtype=psi.dtype)
        return HaltOutput(logits=logits, abg=abg, p_halt=p_halt, p_yes=p_yes)


# ── Pondering ────────────────────────────────────────────────────────────────

@dataclass
class PonderState:
    """Per-batch ponder accumulator (ACT-style)."""

    cumulative_halt: torch.Tensor      # [B, H] in [0, 1] -- cumulative halting mass
    remainder: torch.Tensor            # [B, H] -- 1 - cumulative_halt before final step
    n_iter: torch.Tensor               # [B, H] long -- iterations actually used
    cost: torch.Tensor                 # [B] scalar -- ponder cost contribution
    halted: torch.Tensor               # [B, H] bool -- per-(b,h) halted flag


def init_ponder_state(
    batch_size: int,
    n_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> PonderState:
    return PonderState(
        cumulative_halt=torch.zeros(batch_size, n_heads, device=device, dtype=dtype),
        remainder=torch.ones(batch_size, n_heads, device=device, dtype=dtype),
        n_iter=torch.zeros(batch_size, n_heads, device=device, dtype=torch.long),
        cost=torch.zeros(batch_size, device=device, dtype=dtype),
        halted=torch.zeros(batch_size, n_heads, device=device, dtype=torch.bool),
    )


def update_ponder(
    state: PonderState,
    halt_out: HaltOutput,
    halt_threshold: float = 0.99,
    is_last_iter: bool = False,
) -> Tuple[PonderState, torch.Tensor]:
    r"""Advance ACT-style pondering by one step.

    Returns
    -------
    new_state : updated :class:`PonderState`
    weight    : ``[B, H]`` -- the weight to apply to *this* iteration's psi
                when forming the running mixture of intermediate states.
    """
    p_halt = halt_out.p_halt                                       # [B, H] in [0, 1]
    not_halted = (~state.halted).float()

    # Tentatively add this step's halt mass.
    new_cum = state.cumulative_halt + p_halt * not_halted
    will_halt = (new_cum >= halt_threshold) | torch.tensor(is_last_iter)
    will_halt = will_halt.to(state.halted.dtype)

    # On the halting step, attribute all remaining mass (the "remainder")
    # to this iteration; otherwise use this iteration's halt prob.
    weight = torch.where(
        will_halt.bool() & (~state.halted),
        state.remainder,
        p_halt * not_halted,
    )

    new_halted = state.halted | (will_halt.bool() & (~state.halted))
    new_remainder = torch.where(new_halted, torch.zeros_like(state.remainder),
                                state.remainder - p_halt)
    new_remainder = new_remainder.clamp_min(0.0)
    new_n_iter = state.n_iter + (~state.halted).long()

    # Ponder cost: sum of remainder + n_iter pieces (matches ACT formulation).
    step_cost = (p_halt * not_halted).sum(dim=-1)                  # [B]
    new_cost = state.cost + step_cost

    return PonderState(
        cumulative_halt=new_cum,
        remainder=new_remainder,
        n_iter=new_n_iter,
        cost=new_cost,
        halted=new_halted,
    ), weight
