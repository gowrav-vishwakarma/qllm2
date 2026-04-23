"""Unit tests for SasakiProjectionMemory.

Run on Mac/CPU before any GPU spend:
    uv run python -m pytest v8/qlc/tests/test_projector.py -q
"""

from __future__ import annotations

import math
from typing import Optional

import pytest
import torch

from v8.qlc.projector import SasakiProjectionMemory, SPMState


# ── Helpers ──────────────────────────────────────────────────────────────────

def random_complex(*shape, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Random tensor in split-real complex form."""
    return torch.randn(*shape, 2, generator=generator)


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """Split-real -> torch.complex64 (for tests only; never trained on)."""
    return torch.view_as_complex(x.contiguous())


# ── Tests ────────────────────────────────────────────────────────────────────

def test_init_state_shape_and_dtype():
    spm = SasakiProjectionMemory(dim=16, rank=4, n_heads=2)
    state = spm.init_state(batch_size=3, device=torch.device("cpu"), dtype=torch.float32)
    assert state.U.shape == (3, 2, 16, 4, 2)
    assert state.V.shape == (3, 2, 16, 4, 2)
    assert state.next_slot.shape == (3, 2)
    assert (state.next_slot == 0).all()
    assert (state.filled == 0).all()


def test_orthonormality_after_streaming_writes():
    """After enough streaming writes, U should have orthonormal columns."""
    torch.manual_seed(0)
    spm = SasakiProjectionMemory(dim=16, rank=4, n_heads=1)
    state = spm.init_state(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

    for _ in range(20):
        k = random_complex(2, 1, 16)
        v = random_complex(2, 1, 16)
        gamma = torch.full((2, 1), 0.99)
        state = spm.streaming_step(state, k, v, gamma)

    err = spm.orthonormality_error(state).item()
    assert err < 1e-4, f"Orthonormality drifted: |U^H U - I|_F = {err:.2e}"


def test_retrieve_returns_projection_when_V_eq_U():
    """If V_aligned = U then retrieve(q) = U U^H q = projector applied to q.

    With orthonormal U, retrieve(q) is therefore the orthogonal projection of
    q onto the span of U. Norm of result should be <= norm of q.
    """
    torch.manual_seed(1)
    spm = SasakiProjectionMemory(dim=8, rank=3, n_heads=1)
    # Construct U with explicit orthonormal columns via a fresh QR.
    raw = random_complex(1, 1, 8, 3)
    Uc = to_complex(raw)
    Q, _ = torch.linalg.qr(Uc, mode="reduced")
    U = torch.view_as_real(Q).contiguous()
    state = spm.build_from_basis(U, V_in=U)

    q = random_complex(1, 1, 8)
    y = spm.retrieve(state, q)

    # Apply the projector twice -> same result (idempotence of P).
    state2 = spm.build_from_basis(U, V_in=U)
    y2 = spm.retrieve(state2, y)
    assert torch.allclose(y, y2, atol=1e-5), "Projector not idempotent"

    # Norm of projection <= norm of q.
    n_q = (q[..., 0].square() + q[..., 1].square()).sum().sqrt()
    n_y = (y[..., 0].square() + y[..., 1].square()).sum().sqrt()
    assert n_y <= n_q + 1e-5


def test_retrieve_full_rank_recovers_q():
    """If U is full-rank (r = d) and V_aligned = U, then U U^H = I -> y = q."""
    torch.manual_seed(2)
    spm = SasakiProjectionMemory(dim=4, rank=4, n_heads=1)
    raw = random_complex(1, 1, 4, 4)
    Q, _ = torch.linalg.qr(to_complex(raw), mode="reduced")
    U = torch.view_as_real(Q).contiguous()
    state = spm.build_from_basis(U, V_in=U)

    q = random_complex(1, 1, 4)
    y = spm.retrieve(state, q)
    assert torch.allclose(y, q, atol=1e-5), \
        f"Full-rank retrieve should recover q; max err = {(y - q).abs().max().item():.2e}"


def test_grad_flows_through_streaming_step():
    """The streaming update should be fully differentiable wrt k, v inputs."""
    torch.manual_seed(3)
    spm = SasakiProjectionMemory(dim=8, rank=2, n_heads=1)
    state = spm.init_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    k = random_complex(1, 1, 8).requires_grad_(True)
    v = random_complex(1, 1, 8).requires_grad_(True)
    gamma = torch.tensor([[0.9]])

    state2 = spm.streaming_step(state, k, v, gamma)
    q = random_complex(1, 1, 8)
    y = spm.retrieve(state2, q)
    loss = y.square().sum()
    loss.backward()

    assert k.grad is not None and torch.isfinite(k.grad).all(), "Bad gradient on k"
    assert v.grad is not None and torch.isfinite(v.grad).all(), "Bad gradient on v"
    assert k.grad.abs().max() > 0, "Zero gradient on k"
    assert v.grad.abs().max() > 0, "Zero gradient on v"


def test_circular_slot_eviction():
    """After r writes, slot 0 should be overwritten by write r+1."""
    spm = SasakiProjectionMemory(dim=8, rank=3, n_heads=1)
    state = spm.init_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    # Write 3 deterministic basis-like inputs; check filled count.
    for i in range(3):
        e = torch.zeros(1, 1, 8, 2)
        e[0, 0, i, 0] = 1.0
        v = e.clone()
        state = spm.streaming_step(state, e, v)
    assert (state.filled == 3).all()
    assert (state.next_slot == 0).all()  # wrapped

    # 4th write should also be accepted (slot 0 evicted).
    e = torch.zeros(1, 1, 8, 2); e[0, 0, 3, 0] = 1.0
    state2 = spm.streaming_step(state, e, e)
    assert (state2.filled == 3).all()
    assert (state2.next_slot == 1).all()


def test_chunked_train_is_sequential_streaming():
    """chunked_train should match T sequential streaming_step calls."""
    torch.manual_seed(4)
    spm = SasakiProjectionMemory(dim=6, rank=2, n_heads=1)
    T = 5
    k_seq = random_complex(1, 1, T, 6)
    v_seq = random_complex(1, 1, T, 6)
    gamma_seq = torch.full((1, 1, T), 0.95)

    state_a = spm.init_state(1, torch.device("cpu"), torch.float32)
    ys_a = []
    for t in range(T):
        state_a = spm.streaming_step(state_a, k_seq[:, :, t], v_seq[:, :, t], gamma_seq[..., t])
        ys_a.append(spm.retrieve(state_a, k_seq[:, :, t]))
    ys_a = torch.stack(ys_a, dim=2)

    state_b = spm.init_state(1, torch.device("cpu"), torch.float32)
    ys_b, _ = spm.chunked_train(state_b, k_seq, v_seq, gamma_seq)

    assert torch.allclose(ys_a, ys_b, atol=1e-6)


def test_reorthonormalize_fixes_drift():
    """Manually inject drift, then verify the QR refresh restores orthonormality."""
    torch.manual_seed(5)
    spm = SasakiProjectionMemory(dim=8, rank=4, n_heads=1, qr_refresh_every=0)
    state = spm.init_state(1, torch.device("cpu"), torch.float32)
    # Inject drift by replacing U with a random non-orthonormal matrix.
    state = SPMState(
        U=random_complex(1, 1, 8, 4) * 0.5,
        V=state.V, next_slot=state.next_slot, filled=state.filled,
    )
    err_before = spm.orthonormality_error(state).item()
    assert err_before > 0.1, f"Need real drift to test refresh; got {err_before:.2e}"

    state_fixed = spm._reorthonormalize(state)
    err_after = spm.orthonormality_error(state_fixed).item()
    assert err_after < 1e-4, f"QR refresh failed; err {err_after:.2e}"


def test_quantale_off_symmetrize_breaks_idempotence():
    """sasaki_apply with symmetrize=True should NOT be idempotent (quantale-off)."""
    torch.manual_seed(6)
    spm = SasakiProjectionMemory(dim=6, rank=2, n_heads=1)
    raw = random_complex(1, 1, 6, 2)
    Q, _ = torch.linalg.qr(to_complex(raw), mode="reduced")
    U = torch.view_as_real(Q).contiguous()
    state = spm.build_from_basis(U, V_in=U)

    psi = random_complex(1, 1, 6)
    once = spm.sasaki_apply(state, psi, symmetrize=True)
    twice = spm.sasaki_apply(state, once, symmetrize=True)
    # 0.5(P psi + psi) is generally NOT a fixed point of itself; difference
    # should be detectable.
    diff = (twice - once).abs().max().item()
    assert diff > 1e-4, "Symmetrized Sasaki should not collapse to a fixed point in 1 step"
