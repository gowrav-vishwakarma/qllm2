"""Unit tests for OrthoHalt and ACT pondering."""

from __future__ import annotations

import pytest
import torch

from v8.qlc.halt import (
    OrthoHalt, MLPHalt, init_ponder_state, update_ponder,
)


def random_complex(*shape) -> torch.Tensor:
    return torch.randn(*shape, 2)


def test_orthohalt_output_shapes():
    head = OrthoHalt(dim=8, n_heads=2)
    psi = random_complex(4, 8)
    out = head(psi)
    assert out.logits.shape == (4, 2, 3)
    assert out.abg.shape == (4, 2, 3)
    assert out.p_halt.shape == (4, 2)
    assert out.p_yes.shape == (4, 2)


def test_orthohalt_abg_nonneg_and_bounded():
    head = OrthoHalt(dim=8, n_heads=1)
    psi = random_complex(4, 8) * 0.1  # small psi -> small alpha+beta
    out = head(psi)
    assert (out.abg >= 0).all(), "alpha/beta/gamma must be non-negative"
    # gamma must be in [0, 1].
    assert (out.abg[..., 2] <= 1.0 + 1e-5).all()


def test_orthohalt_alpha_plus_beta_eq_psi_norm_sq():
    """alpha + beta should equal ||psi||^2 (when target u is unit norm)."""
    torch.manual_seed(0)
    head = OrthoHalt(dim=8, n_heads=1)
    psi = random_complex(3, 8)
    u = head.target_vector(psi)                    # [B, H, d, 2]
    psi_h = psi.unsqueeze(1)                       # [B, 1, d, 2]
    abg = head.abg(psi_h, u)                       # [B, 1, 3]
    psi_norm_sq = (psi[..., 0].square() + psi[..., 1].square()).sum(dim=-1)  # [B]
    assert torch.allclose(abg[..., 0] + abg[..., 1], psi_norm_sq.unsqueeze(-1), atol=1e-4)


def test_orthohalt_target_unit_norm():
    head = OrthoHalt(dim=8, n_heads=2)
    psi = random_complex(4, 8)
    u = head.target_vector(psi)
    norms = (u[..., 0].square() + u[..., 1].square()).sum(dim=-1).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_orthohalt_grad_flows():
    head = OrthoHalt(dim=8, n_heads=1)
    psi = random_complex(2, 8).requires_grad_(True)
    out = head(psi)
    (out.p_yes.mean() + out.abg.sum() * 0.01).backward()
    assert psi.grad is not None and torch.isfinite(psi.grad).all()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"No grad on {name}"


def test_mlp_halt_signature_compat():
    """MLPHalt must produce the same output shapes as OrthoHalt."""
    o = OrthoHalt(dim=8, n_heads=2)
    m = MLPHalt(dim=8, n_heads=2)
    psi = random_complex(3, 8)
    a = o(psi); b = m(psi)
    assert a.logits.shape == b.logits.shape
    assert a.abg.shape == b.abg.shape
    assert a.p_halt.shape == b.p_halt.shape
    assert a.p_yes.shape == b.p_yes.shape


def test_ponder_eventually_halts_at_last_iter():
    """If continue prob is large but is_last_iter triggers, we must halt."""
    state = init_ponder_state(2, 1, torch.device("cpu"), torch.float32)
    halt_out = type("HO", (), {})()
    halt_out.p_halt = torch.tensor([[0.01], [0.01]])
    state, w1 = update_ponder(state, halt_out, halt_threshold=0.99, is_last_iter=False)
    assert (state.halted == False).all()
    state, w2 = update_ponder(state, halt_out, halt_threshold=0.99, is_last_iter=True)
    assert state.halted.all()
    # On the forced final step, weight should equal the remainder before the step.
    assert torch.all(w2 > 0.9)


def test_ponder_stops_when_threshold_exceeded():
    state = init_ponder_state(1, 1, torch.device("cpu"), torch.float32)
    halt_out = type("HO", (), {})()
    halt_out.p_halt = torch.tensor([[0.999]])
    state, w = update_ponder(state, halt_out, halt_threshold=0.5, is_last_iter=False)
    assert state.halted.all()
    # weight = remainder (which was 1.0 going in)
    assert torch.allclose(w, torch.ones_like(w), atol=1e-3)
