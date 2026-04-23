"""Unit tests for EffectAlgebraBank."""

from __future__ import annotations

import pytest
import torch

from v8.qlc.effect_bank import EffectAlgebraBank
from v8.qlc.projector import SasakiProjectionMemory


def random_complex(*shape) -> torch.Tensor:
    return torch.randn(*shape, 2)


def test_gates_in_unit_interval():
    bank = EffectAlgebraBank(dim=8, bank_size=16, n_heads=1)
    g = bank.gates()
    assert (g >= 0).all() and (g <= 1).all()


def test_effect_directions_are_unit_norm():
    bank = EffectAlgebraBank(dim=8, bank_size=16, n_heads=1)
    u = bank.effect_directions()
    norms = (u[..., 0].square() + u[..., 1].square()).sum(dim=-1).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_probe_scores_nonnegative():
    torch.manual_seed(0)
    bank = EffectAlgebraBank(dim=8, bank_size=16, n_heads=2)
    psi = random_complex(3, 8)
    scores = bank.probe(psi)
    assert scores.shape == (3, 2, 16)
    assert (scores >= 0).all()


def test_select_top_k_returns_orthonormal_basis():
    torch.manual_seed(1)
    bank = EffectAlgebraBank(dim=8, bank_size=32, n_heads=1)
    psi = random_complex(4, 8)
    U, V = bank.select_top_k(psi, k=4, rank=4, reason_heads=1)
    assert U.shape == (4, 1, 8, 4, 2)
    assert V.shape == (4, 1, 8, 4, 2)

    # Check U^H U = I per (b, h)
    Uc = torch.view_as_complex(U.contiguous())
    gram = Uc.transpose(-1, -2).conj() @ Uc
    eye = torch.eye(4, dtype=gram.dtype, device=gram.device).expand_as(gram)
    diff = (gram - eye).abs().max().item()
    assert diff < 1e-4, f"Top-k basis not orthonormal: max err {diff:.2e}"


def test_select_top_k_pads_when_k_lt_rank():
    """When k < rank, the returned U should still have `rank` columns (padded)."""
    torch.manual_seed(2)
    bank = EffectAlgebraBank(dim=8, bank_size=32, n_heads=1)
    psi = random_complex(2, 8)
    U, V = bank.select_top_k(psi, k=2, rank=4, reason_heads=1)
    assert U.shape == (2, 1, 8, 4, 2)
    # Last (rank - k) = 2 columns should be zero.
    Uc = torch.view_as_complex(U.contiguous())
    pad = Uc[..., 2:]
    assert pad.abs().max().item() < 1e-6


def test_select_top_k_works_with_spm_build_from_basis():
    torch.manual_seed(3)
    bank = EffectAlgebraBank(dim=8, bank_size=32, n_heads=1)
    spm = SasakiProjectionMemory(dim=8, rank=4, n_heads=1)
    psi = random_complex(2, 8)
    U, V = bank.select_top_k(psi, k=4, rank=4, reason_heads=1)
    state = spm.build_from_basis(U, V_in=V)
    err = spm.orthonormality_error(state).item()
    assert err < 1e-4


def test_grad_flows_to_bank_through_topk_and_retrieve():
    torch.manual_seed(4)
    bank = EffectAlgebraBank(dim=8, bank_size=16, n_heads=1)
    spm = SasakiProjectionMemory(dim=8, rank=4, n_heads=1)
    psi = random_complex(2, 8).requires_grad_(True)
    U, V = bank.select_top_k(psi, k=4, rank=4, reason_heads=1)
    state = spm.build_from_basis(U, V_in=V)
    psi_h = psi.unsqueeze(1)  # [B, H=1, d, 2]
    y = spm.retrieve(state, psi_h)
    y.sum().backward()
    assert psi.grad is not None
    assert torch.isfinite(psi.grad).all()
    # At least one of the bank parameters should accumulate gradient.
    grads = [
        bank.u_real.grad, bank.u_imag.grad,
        bank.w_real.grad, bank.w_imag.grad,
        bank.s_logit.grad,
    ]
    assert any(g is not None and g.abs().max() > 0 for g in grads), \
        "No gradient flowed back to any bank parameter"


def test_infonce_loss_decreases_with_correct_routing():
    """Sanity: gradient on bank should reduce InfoNCE loss in a couple of steps."""
    torch.manual_seed(5)
    bank = EffectAlgebraBank(dim=8, bank_size=16, n_heads=1)
    opt = torch.optim.SGD(bank.parameters(), lr=0.5)

    B, N = 4, 3
    psi_pos = random_complex(B, 8)
    psi_neg = random_complex(B, N, 8)
    gold = torch.randint(0, 16, (B,))

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = bank.infonce_loss(psi_pos, psi_neg, gold, temperature=0.5)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] - 0.05, \
        f"InfoNCE didn't decrease: start={losses[0]:.3f} end={losses[-1]:.3f}"
