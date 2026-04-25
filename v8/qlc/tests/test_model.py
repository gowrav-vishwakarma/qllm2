"""End-to-end smoke tests for V8LM."""

from __future__ import annotations

import pytest
import torch

from v8.config import get_config, V8Config, QLCConfig
from v8.model import V8LM
from v8.backbone import V7Config


@pytest.fixture(scope="module")
def tiny_v8_passthrough():
    cfg = V8Config(
        backbone=V7Config(
            vocab_size=64, dim=16, n_heads=2, head_dim=8,
            n_layers=2, expand=2, dropout=0.0, max_seq_len=64,
            hierarchical_dt=False, cross_level=False,
            chunk_size=0, multi_scale_loss=False, use_reverse_assoc=False,
            gradient_checkpointing=False,
        ),
        qlc=QLCConfig(enabled=False),
        stage="A",
    )
    return V8LM(cfg)


@pytest.fixture(scope="module")
def tiny_v8_qlc():
    cfg = V8Config(
        backbone=V7Config(
            vocab_size=64, dim=16, n_heads=2, head_dim=8,
            n_layers=2, expand=2, dropout=0.0, max_seq_len=64,
            hierarchical_dt=False, cross_level=False,
            chunk_size=0, multi_scale_loss=False, use_reverse_assoc=False,
            gradient_checkpointing=False,
        ),
        qlc=QLCConfig(
            enabled=True, rank=4, bank_size=32, top_k=2, t_max=2,
            n_heads=1, ponder_lambda=0.01,
        ),
        stage="B", freeze_backbone=False,
    )
    return V8LM(cfg)


def test_passthrough_forward_shape(tiny_v8_passthrough):
    model = tiny_v8_passthrough
    x = torch.randint(0, 64, (2, 8))
    logits, states, aux = model(x)
    assert logits.shape == (2, 8, 64)
    assert len(states) == 2
    assert aux.dim() == 0
    assert aux.item() == 0.0


def test_qlc_forward_shape(tiny_v8_qlc):
    model = tiny_v8_qlc
    x = torch.randint(0, 64, (2, 8))
    logits, states, aux = model(x)
    assert logits.shape == (2, 8, 64)
    assert aux.item() >= 0.0


def test_freeze_backbone_only_qlc_is_trainable():
    cfg = V8Config(
        backbone=V7Config(
            vocab_size=64, dim=16, n_heads=2, head_dim=8,
            n_layers=2, expand=2, dropout=0.0, max_seq_len=64,
            hierarchical_dt=False, cross_level=False,
            chunk_size=0, multi_scale_loss=False, use_reverse_assoc=False,
            gradient_checkpointing=False,
        ),
        qlc=QLCConfig(enabled=True, rank=4, bank_size=32, top_k=2, t_max=2),
        stage="B", freeze_backbone=True,
    )
    model = V8LM(cfg)
    backbone_total = (
        sum(p.numel() for p in model.embed.parameters())
        + sum(p.numel() for p in model.embed_norm.parameters())
        + sum(p.numel() for p in model.blocks.parameters())
        + sum(p.numel() for p in model.output_norm.parameters())
        + sum(p.numel() for p in model.lm_head_proj.parameters())
        + sum(p.numel() for p in model.lm_head_norm.parameters())
    )
    qlc_total = sum(p.numel() for p in model.qlc.parameters())
    assert model.trainable_parameters() == qlc_total
    assert model.trainable_parameters() < backbone_total


def test_passthrough_grad_flows(tiny_v8_passthrough):
    model = tiny_v8_passthrough
    x = torch.randint(0, 64, (2, 8))
    logits, _, _ = model(x)
    logits.sum().backward()
    # Embedding should have grad.
    assert model.embed.embed_real.weight.grad is not None
    # Clear grads so the fixture stays fresh.
    model.zero_grad(set_to_none=True)


def test_qlc_grad_flows_to_bank(tiny_v8_qlc):
    model = tiny_v8_qlc
    x = torch.randint(0, 64, (2, 8))
    logits, _, aux = model(x)
    (logits.sum() + aux).backward()
    assert model.qlc.bank.u_real.grad is not None
    assert model.qlc.bank.u_real.grad.abs().max() > 0
    model.zero_grad(set_to_none=True)


def test_backbone_logits_matches_qlc_disabled(tiny_v8_passthrough):
    """For a passthrough model, backbone_logits and forward must agree."""
    model = tiny_v8_passthrough
    model.eval()
    x = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        a, _, _ = model(x)
        b = model.backbone_logits(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_count_parameters_includes_qlc(tiny_v8_qlc, tiny_v8_passthrough):
    p_with = tiny_v8_qlc.count_parameters()
    p_without = tiny_v8_passthrough.count_parameters()
    assert p_with["qlc"] > 0
    assert p_without["qlc"] == 0


def test_interleaved_qlc_forward_and_diagnostics():
    cfg = V8Config(
        backbone=V7Config(
            vocab_size=64, dim=16, n_heads=2, head_dim=8,
            n_layers=3, expand=2, dropout=0.0, max_seq_len=64,
            hierarchical_dt=False, cross_level=False,
            chunk_size=0, multi_scale_loss=False, use_reverse_assoc=False,
            gradient_checkpointing=False,
        ),
        qlc=QLCConfig(
            enabled=True, rank=4, bank_size=32, top_k=2, t_max=2,
            n_heads=1, ponder_lambda=0.01,
            use_iteration_context=True, use_layer_context=True,
            rope_conditioned_probe=True, weighted_projector=True,
            renormalize_psi=False,
        ),
        stage="A",
        qlc_insert_layers=(0, 1),
        shared_interleaved_qlc=True,
        final_qlc_enabled=False,
    )
    model = V8LM(cfg)
    x = torch.randint(0, 64, (2, 8))
    logits, states, aux = model(x, return_qlc_diag=True)
    assert logits.shape == (2, 8, 64)
    assert len(states) == 3
    assert aux.item() >= 0.0
    diags = model.last_qlc_diagnostics_all()
    assert [d.insertion_point for d in diags] == ["layer_0", "layer_1"]
    assert diags[0].memory_echo_cos is None
    assert diags[1].memory_echo_cos is not None
    assert model.primary_qlc() is model.interleaved_qlc
