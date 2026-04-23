"""Unit tests for QuantumLogicCore reasoning loop."""

from __future__ import annotations

import pytest
import torch

from v8.qlc.reason_loop import QuantumLogicCore


def random_complex(*shape) -> torch.Tensor:
    return torch.randn(*shape, 2)


def test_qlc_forward_shapes():
    qlc = QuantumLogicCore(dim=8, rank=4, bank_size=16, top_k=4, t_max=2, n_heads=1)
    psi = random_complex(2, 5, 8)
    out, cost, diag = qlc(psi, return_diagnostics=True)
    assert out.shape == psi.shape
    assert cost.dim() == 0  # scalar
    assert diag is not None
    assert diag.mean_iter > 0


def test_qlc_grad_flows_to_bank_and_halt():
    torch.manual_seed(0)
    qlc = QuantumLogicCore(dim=8, rank=4, bank_size=16, top_k=4, t_max=2, n_heads=1)
    psi = random_complex(2, 3, 8).requires_grad_(True)
    out, cost, _ = qlc(psi)
    loss = out.square().mean() + 0.01 * cost
    loss.backward()
    assert psi.grad is not None and torch.isfinite(psi.grad).all()
    # Sample a few module params and check they got grads.
    seen_grad = 0
    for p in qlc.parameters():
        if p.grad is not None and p.grad.abs().max() > 0:
            seen_grad += 1
    assert seen_grad >= 3, f"Only {seen_grad} params received non-zero grads"


def test_qlc_passthrough_when_out_scale_zero():
    """With out_scale forced to 0, QLC output must equal the input exactly."""
    qlc = QuantumLogicCore(dim=8, rank=4, bank_size=16, top_k=4, t_max=2, n_heads=1)
    with torch.no_grad():
        qlc.out_scale.zero_()
    psi = random_complex(2, 4, 8)
    out, _, _ = qlc(psi)
    assert torch.allclose(out, psi, atol=1e-6)


def test_qlc_quantale_off_changes_output():
    """The V8-F ablation must produce a different output than the default."""
    torch.manual_seed(1)
    psi = random_complex(2, 3, 8)
    qlc_on = QuantumLogicCore(dim=8, rank=4, bank_size=16, top_k=4, t_max=2,
                              quantale_off=False)
    torch.manual_seed(2)  # so the bank/halt params differ deterministically
    qlc_off = QuantumLogicCore(dim=8, rank=4, bank_size=16, top_k=4, t_max=2,
                               quantale_off=True)
    # Same params except quantale_off: copy state dict.
    qlc_off.load_state_dict(qlc_on.state_dict())
    out_on, _, _ = qlc_on(psi)
    out_off, _, _ = qlc_off(psi)
    assert not torch.allclose(out_on, out_off, atol=1e-4), \
        "quantale_off had no effect on output -- ablation switch is dead"


def test_qlc_orthohalt_off_uses_mlp_halt():
    qlc_on = QuantumLogicCore(dim=8, rank=4, bank_size=16, top_k=4, t_max=2,
                              orthohalt_off=False)
    qlc_off = QuantumLogicCore(dim=8, rank=4, bank_size=16, top_k=4, t_max=2,
                               orthohalt_off=True)
    from v8.qlc.halt import OrthoHalt, MLPHalt
    assert isinstance(qlc_on.halt, OrthoHalt)
    assert isinstance(qlc_off.halt, MLPHalt)


def test_qlc_t_max_one_runs_single_iteration():
    qlc = QuantumLogicCore(dim=8, rank=4, bank_size=16, top_k=4, t_max=1)
    psi = random_complex(2, 3, 8)
    _, _, diag = qlc(psi, return_diagnostics=True)
    assert diag.mean_iter == 1.0


def test_qlc_multi_head_averages_outputs():
    qlc = QuantumLogicCore(dim=8, rank=4, bank_size=16, top_k=4, t_max=2, n_heads=3)
    psi = random_complex(2, 3, 8)
    out, _, _ = qlc(psi)
    assert out.shape == psi.shape


def test_qlc_handles_seq_len_one():
    qlc = QuantumLogicCore(dim=8, rank=4, bank_size=16, top_k=4, t_max=2)
    psi = random_complex(3, 1, 8)
    out, _, _ = qlc(psi)
    assert out.shape == (3, 1, 8, 2)
