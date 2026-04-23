r"""QuantumLogicCore: the iterative reasoning loop.

For each token's hidden state :math:`\psi_0` from the QPAM backbone, the QLC
runs up to ``T_max`` iterations of

  1. **Probe** the effect bank for the top-k effects matching :math:`\psi_i`.
  2. **Build** a rank-r facts projector :math:`\Pi_F` from the selected
     effects (orthonormalized via QR; see
     :class:`EffectAlgebraBank.select_top_k`).
  3. **Sasaki update**: :math:`\psi_{i+1} = \Pi_F \psi_i` (or the symmetrized
     half-step in the ``quantale_off`` ablation).
  4. **OrthoHalt** reads :math:`(\alpha, \beta, \gamma)` from
     :math:`\psi_{i+1}` against a learned target effect, decides whether to
     halt or continue.
  5. ACT-style pondering accumulates a soft mixture of :math:`\psi_i` across
     iterations weighted by per-step halt mass.

The output is the pondered state :math:`\psi_M` ready for the LM head, plus
the ponder cost (training regularizer) and bookkeeping diagnostics
(``mean_iter``, ``mean_alpha``, ``mean_beta``, ``mean_gamma``,
``halt_distribution``).

Operates *per token position* in parallel: the loop iterates over T_max
(typically 1-4), not over the sequence length T. The full backbone hidden
sequence is processed by reshaping to ``[B*T, d, 2]``.

Heads
-----

The loop's "reasoning heads" are independent of the backbone PAM heads. The
default ``n_heads=1`` keeps params minimal; multi-head reasoning lets each
head specialize on a different effect dictionary slice. When ``n_heads>1``
the per-head outputs are averaged before the LM head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v8.qlc.projector import SasakiProjectionMemory
from v8.qlc.effect_bank import EffectAlgebraBank
from v8.qlc.halt import (
    OrthoHalt, MLPHalt, init_ponder_state, update_ponder, HaltOutput,
)


@dataclass
class QLCDiagnostics:
    """Per-batch diagnostics for the reasoning loop.

    All scalar fields are means across (B, H) unless noted otherwise.
    """

    mean_iter: float
    mean_alpha: float
    mean_beta: float
    mean_gamma: float
    halt_yes_rate: float
    halt_no_rate: float
    continue_rate: float
    n_iters_per_sample: torch.Tensor   # [B, H] long
    ponder_cost: torch.Tensor          # [B] scalar contribution


class QuantumLogicCore(nn.Module):
    r"""Iterative reasoning core sitting between the QPAM backbone and the LM head.

    Parameters
    ----------
    dim : int
        Complex dimension of the backbone hidden state.
    rank : int
        Rank :math:`r` of the per-iteration facts projector :math:`\Pi_F`.
    bank_size : int
        Number of effects in the :class:`EffectAlgebraBank`.
    top_k : int
        How many effects feed :math:`\Pi_F` per iteration.
    t_max : int
        Maximum reasoning iterations.
    n_heads : int
        Number of independent reasoning heads.
    ponder_lambda : float
        ACT-style ponder cost coefficient (used by the trainer; this module
        only *returns* the cost, it does not weight it into the LM loss).
    bank_temperature : float
        Temperature for the bank probe softmax (lower -> sharper top-k).
    quantale_off : bool
        Ablation V8-F. If True, replace Sasaki update :math:`\Pi_F \psi` with
        :math:`\frac{1}{2}(\Pi_F \psi + \psi)` -- the symmetrized form whose
        composition is commutative and therefore loses the quantale ordering.
    orthohalt_off : bool
        Ablation V8-G. If True, replace :class:`OrthoHalt` with a plain
        :class:`MLPHalt` head that never sees ``(alpha, beta, gamma)``.
    qr_refresh_every : int
        Re-orthonormalize the projector basis every K iterations (0 = never).
    halt_threshold : float
        ACT halting threshold on cumulative halt mass.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 8,
        bank_size: int = 2048,
        top_k: int = 4,
        t_max: int = 4,
        n_heads: int = 1,
        ponder_lambda: float = 0.01,
        bank_temperature: float = 1.0,
        quantale_off: bool = False,
        orthohalt_off: bool = False,
        qr_refresh_every: int = 0,
        halt_threshold: float = 0.99,
    ):
        super().__init__()
        if top_k > bank_size:
            raise ValueError(f"top_k ({top_k}) > bank_size ({bank_size})")
        if rank > dim:
            raise ValueError(f"rank ({rank}) > dim ({dim})")
        self.dim = dim
        self.rank = rank
        self.bank_size = bank_size
        self.top_k = top_k
        self.t_max = max(1, t_max)
        self.n_heads = n_heads
        self.ponder_lambda = ponder_lambda
        self.bank_temperature = bank_temperature
        self.quantale_off = quantale_off
        self.orthohalt_off = orthohalt_off
        self.halt_threshold = halt_threshold

        self.bank = EffectAlgebraBank(
            dim=dim, bank_size=bank_size, n_heads=n_heads,
        )
        self.spm = SasakiProjectionMemory(
            dim=dim, rank=rank, n_heads=n_heads, qr_refresh_every=qr_refresh_every,
        )
        if orthohalt_off:
            self.halt = MLPHalt(dim=dim, n_heads=n_heads)
        else:
            self.halt = OrthoHalt(dim=dim, n_heads=n_heads)

        # Multi-head merge: when n_heads > 1 we average the per-head pondered
        # states. A learnable per-head mix logits gives the model freedom to
        # weight heads differently.
        self.head_mix = nn.Parameter(torch.zeros(n_heads))

        # Output residual scale: starts small so the QLC contribution is a
        # small perturbation on the backbone (helps Stage B learn smoothly
        # against a frozen backbone).
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        psi: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[QLCDiagnostics]]:
        r"""Run the reasoning loop on ``psi: [B, T, d, 2]``.

        Returns
        -------
        psi_out : ``[B, T, d, 2]`` -- pondered state to feed the LM head.
        ponder_cost : scalar -- :math:`\sum_t p_\text{halt}^{(t)}` averaged over
                      batch+heads (the trainer multiplies by ``ponder_lambda``).
        diag : optional :class:`QLCDiagnostics` -- when ``return_diagnostics=True``.
        """
        B, T, d, two = psi.shape
        H = self.n_heads
        assert d == self.dim, f"psi dim {d} != QLC dim {self.dim}"

        # Flatten (B, T) so the loop runs over independent token positions.
        psi_flat = psi.reshape(B * T, d, two)
        BT = psi_flat.shape[0]

        # ACT-style pondering accumulator.
        ponder = init_ponder_state(BT, H, psi.device, psi.dtype)

        # Pondered output accumulator -- per head [BT, H, d, 2] -- gradients
        # flow through the soft mixture.
        psi_out_acc = torch.zeros(BT, H, d, two, device=psi.device, dtype=psi.dtype)

        # Diagnostic accumulators.
        sum_alpha = torch.zeros((), device=psi.device, dtype=psi.dtype)
        sum_beta = torch.zeros_like(sum_alpha)
        sum_gamma = torch.zeros_like(sum_alpha)
        n_yes = torch.zeros_like(sum_alpha)
        n_no = torch.zeros_like(sum_alpha)
        n_cont = torch.zeros_like(sum_alpha)
        diag_count = 0

        # Per-head iteration: we keep psi_per_head separate for each head so
        # different heads can take different reasoning trajectories.
        psi_h = psi_flat.unsqueeze(1).expand(BT, H, d, two).contiguous()

        for it in range(self.t_max):
            is_last = (it == self.t_max - 1)

            # 1-2. Probe + build Pi_F. We pass the *averaged* psi across heads
            # to the bank's probe (the bank's own probe heads then re-split it
            # per head). This keeps the bank parameter count linear in M, not
            # M*H.
            psi_pool = psi_h.mean(dim=1)                  # [BT, d, 2]
            U, V = self.bank.select_top_k(
                psi_pool, k=self.top_k, rank=self.rank,
                reason_heads=H, temperature=self.bank_temperature,
            )                                              # both [BT, H, d, r, 2]
            spm_state = self.spm.build_from_basis(U, V_in=V)

            # 3. Sasaki update per head.
            psi_next = self.spm.sasaki_apply(
                spm_state, psi_h, symmetrize=self.quantale_off,
            )                                              # [BT, H, d, 2]

            # Renormalize psi_next to stay on the unit sphere (avoids the
            # projector shrinking the state to zero when overlap is low,
            # which would starve subsequent iterations of any signal).
            sq = (psi_next[..., 0].square() + psi_next[..., 1].square()).sum(dim=-1, keepdim=True)
            scale = sq.clamp_min(1e-6).sqrt().unsqueeze(-1)
            psi_next = psi_next / scale

            # 4. OrthoHalt readout from the pondered psi (use flattened across heads
            # by averaging again so halt sees the consensus state).
            halt_out = self.halt(psi_next.mean(dim=1))     # HaltOutput, p_halt: [BT, H]
            ponder, weight = update_ponder(
                ponder, halt_out, halt_threshold=self.halt_threshold,
                is_last_iter=is_last,
            )

            # 5. Accumulate weighted contribution.
            psi_out_acc = psi_out_acc + weight.unsqueeze(-1).unsqueeze(-1) * psi_next

            # Roll forward.
            psi_h = psi_next

            # Diagnostics.
            sum_alpha = sum_alpha + halt_out.abg[..., 0].mean()
            sum_beta = sum_beta + halt_out.abg[..., 1].mean()
            sum_gamma = sum_gamma + halt_out.abg[..., 2].mean()
            probs = halt_out.logits.softmax(dim=-1)
            n_yes = n_yes + probs[..., 0].mean()
            n_no = n_no + probs[..., 1].mean()
            n_cont = n_cont + probs[..., 2].mean()
            diag_count += 1

            if ponder.halted.all():
                break

        # Merge per-head: weighted mean using softmax(head_mix).
        head_w = self.head_mix.softmax(dim=0).view(1, H, 1, 1)
        psi_merged = (psi_out_acc * head_w).sum(dim=1)     # [BT, d, 2]

        # Output: backbone state + small residual from QLC.
        psi_residual = psi_merged - psi_flat
        psi_out_flat = psi_flat + self.out_scale * psi_residual

        psi_out = psi_out_flat.view(B, T, d, two)

        ponder_cost = ponder.cost.mean() if ponder.cost.numel() > 0 else torch.tensor(0.0)

        if return_diagnostics:
            inv = 1.0 / max(diag_count, 1)
            diag = QLCDiagnostics(
                mean_iter=ponder.n_iter.float().mean().item(),
                mean_alpha=(sum_alpha * inv).item(),
                mean_beta=(sum_beta * inv).item(),
                mean_gamma=(sum_gamma * inv).item(),
                halt_yes_rate=(n_yes * inv).item(),
                halt_no_rate=(n_no * inv).item(),
                continue_rate=(n_cont * inv).item(),
                n_iters_per_sample=ponder.n_iter.detach().cpu(),
                ponder_cost=ponder.cost.detach().cpu(),
            )
            return psi_out, ponder_cost, diag

        return psi_out, ponder_cost, None

    # ── Convenience: parameter count ───────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, rank={self.rank}, M={self.bank_size}, "
            f"top_k={self.top_k}, T_max={self.t_max}, n_heads={self.n_heads}, "
            f"quantale_off={self.quantale_off}, orthohalt_off={self.orthohalt_off}"
        )
