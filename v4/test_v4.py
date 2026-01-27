#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Test Suite: Validate all components work correctly

Run with: python test_v4.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_phase2d():
    """Test Phase2D operations"""
    print("\n" + "=" * 60)
    print("Testing Phase2D operations...")
    print("=" * 60)
    
    from v4.core.phase2d import (
        phase2d_multiply, phase2d_conjugate, phase2d_magnitude,
        phase2d_normalize, phase2d_apply_iota, phase2d_coherence,
        Phase2DEmbed, Phase2DLinear, Phase2DLayerNorm, IotaBlock
    )
    
    # Test basic operations
    a = torch.randn(2, 4, 8, 2)  # [batch, seq, dim, 2]
    b = torch.randn(2, 4, 8, 2)
    
    # Multiply
    c = phase2d_multiply(a, b)
    assert c.shape == a.shape, f"Multiply shape mismatch: {c.shape}"
    print("  ✓ phase2d_multiply")
    
    # Conjugate
    conj = phase2d_conjugate(a)
    assert conj.shape == a.shape
    assert torch.allclose(conj[..., 0], a[..., 0])  # Real unchanged
    assert torch.allclose(conj[..., 1], -a[..., 1])  # Imag negated
    print("  ✓ phase2d_conjugate")
    
    # Magnitude
    mag = phase2d_magnitude(a)
    assert mag.shape == a.shape[:-1]
    assert (mag >= 0).all()
    print("  ✓ phase2d_magnitude")
    
    # Normalize
    norm = phase2d_normalize(a)
    norm_mag = phase2d_magnitude(norm)
    assert torch.allclose(norm_mag, torch.ones_like(norm_mag), atol=1e-5)
    print("  ✓ phase2d_normalize")
    
    # Apply iota (multiply by i)
    iota_result = phase2d_apply_iota(a)
    assert iota_result.shape == a.shape
    assert torch.allclose(iota_result[..., 0], -a[..., 1])  # real = -imag
    assert torch.allclose(iota_result[..., 1], a[..., 0])   # imag = real
    print("  ✓ phase2d_apply_iota")
    
    # Coherence
    coh = phase2d_coherence(a, b)
    assert coh.shape == a.shape[:-2]  # [batch, seq]
    print("  ✓ phase2d_coherence")
    
    # Test modules
    embed = Phase2DEmbed(vocab_size=100, dim=16)
    token_ids = torch.randint(0, 100, (2, 8))
    embedded = embed(token_ids)
    assert embedded.shape == (2, 8, 16, 2)
    print("  ✓ Phase2DEmbed")
    
    linear = Phase2DLinear(16, 32)
    out = linear(embedded)
    assert out.shape == (2, 8, 32, 2)
    print("  ✓ Phase2DLinear")
    
    norm_layer = Phase2DLayerNorm(16)
    normed = norm_layer(embedded)
    assert normed.shape == embedded.shape
    print("  ✓ Phase2DLayerNorm")
    
    iota_block = IotaBlock(dim=16)
    rotated = iota_block(embedded)
    assert rotated.shape == embedded.shape
    print("  ✓ IotaBlock")
    
    print("\n✅ All Phase2D tests passed!")


def test_banks():
    """Test Phase Banks"""
    print("\n" + "=" * 60)
    print("Testing Phase Banks...")
    print("=" * 60)
    
    from v4.banks import SemanticPhaseBank, ContextPhaseBank, LanguagePhaseBank
    
    batch, seq, dim = 2, 16, 64
    x = torch.randn(batch, seq, dim, 2)
    
    # Semantic bank
    semantic = SemanticPhaseBank(dim=dim, num_concepts=128, num_layers=2)
    out = semantic(x)
    assert out.shape == x.shape, f"Semantic shape: {out.shape}"
    print("  ✓ SemanticPhaseBank")
    
    # Context bank
    context = ContextPhaseBank(dim=dim, window_size=4, num_layers=2)
    out = context(x)
    assert out.shape == x.shape
    print("  ✓ ContextPhaseBank")
    
    # Language bank
    language = LanguagePhaseBank(dim=dim, num_languages=8, num_layers=2)
    out = language(x, context={'language_id': 1})
    assert out.shape == x.shape
    print("  ✓ LanguagePhaseBank")
    
    print("\n✅ All Bank tests passed!")


def test_backbone():
    """Test Backbone"""
    print("\n" + "=" * 60)
    print("Testing Backbone...")
    print("=" * 60)
    
    from v4.backbone import OscillatorySSM
    
    batch, seq, dim = 2, 16, 64
    state_dim = 128
    x = torch.randn(batch, seq, dim, 2)
    
    backbone = OscillatorySSM(dim=dim, state_dim=state_dim, num_layers=4)
    
    # Without state
    out, state = backbone(x)
    assert out.shape == x.shape, f"Backbone output shape: {out.shape}"
    assert state.hidden.shape[1] == batch
    print("  ✓ OscillatorySSM forward")
    
    # With state (streaming)
    x2 = torch.randn(batch, 8, dim, 2)
    out2, state2 = backbone(x2, state=state)
    assert out2.shape == (batch, 8, dim, 2)
    print("  ✓ OscillatorySSM streaming")
    
    print("\n✅ All Backbone tests passed!")


def test_coupler():
    """Test Coupler"""
    print("\n" + "=" * 60)
    print("Testing Coupler...")
    print("=" * 60)
    
    from v4.coupler import InterferenceCoupler
    
    batch, seq, dim = 2, 16, 64
    
    bank_outputs = {
        'semantic': torch.randn(batch, seq, dim, 2),
        'context': torch.randn(batch, seq, dim, 2),
    }
    
    coupler = InterferenceCoupler(dim=dim, bank_names=['semantic', 'context'])
    out = coupler(bank_outputs)
    assert out.shape == (batch, seq, dim, 2)
    print("  ✓ InterferenceCoupler forward")
    
    # Test coupling loss
    loss = coupler.compute_coupling_loss(bank_outputs)
    assert loss is not None
    assert loss.dim() == 0  # scalar
    print("  ✓ InterferenceCoupler coupling_loss")
    
    print("\n✅ All Coupler tests passed!")


def test_memory():
    """Test Memory"""
    print("\n" + "=" * 60)
    print("Testing Memory...")
    print("=" * 60)
    
    from v4.memory import PhaseAssociativeMemory
    
    batch, seq, dim = 2, 16, 64
    num_slots = 128
    
    memory = PhaseAssociativeMemory(dim=dim, num_slots=num_slots)
    
    # Read with sparse mode (default - memory efficient)
    query = torch.randn(batch, seq, dim, 2)
    result = memory.read(query, top_k=32, use_sparse=True)
    assert result.values.shape == (batch, seq, dim, 2)
    assert result.attention is None  # Sparse mode returns None for attention
    print("  ✓ PhaseAssociativeMemory read (sparse)")
    
    # Read with dense mode (for debugging)
    result_dense = memory.read(query, top_k=32, use_sparse=False)
    assert result_dense.values.shape == (batch, seq, dim, 2)
    assert result_dense.attention.shape == (batch, seq, num_slots)
    print("  ✓ PhaseAssociativeMemory read (dense)")
    
    # Test chunked top-k directly
    attn_weights, attn_indices = memory._compute_attention(
        memory.query_proj(query), memory.keys, top_k=32, return_sparse=True
    )
    assert attn_weights.shape == (batch, seq, 32)
    assert attn_indices.shape == (batch, seq, 32)
    print("  ✓ PhaseAssociativeMemory chunked top-k")
    
    # Write
    memory.train()
    key = torch.randn(batch, dim, 2)
    value = torch.randn(batch, dim, 2)
    memory.write(key, value)
    print("  ✓ PhaseAssociativeMemory write")
    
    # Consolidate
    memory.consolidate()
    print("  ✓ PhaseAssociativeMemory consolidate")
    
    print("\n✅ All Memory tests passed!")


def test_objectives():
    """Test Objectives"""
    print("\n" + "=" * 60)
    print("Testing Objectives...")
    print("=" * 60)
    
    from v4.objectives import CrossEntropyObjective, CoherenceObjective, EnergyObjective, CouplingObjective
    
    batch, seq, vocab_size, dim = 2, 16, 1000, 64
    
    model_output = {
        'logits': torch.randn(batch, seq, vocab_size),
        'phase_states': torch.randn(batch, seq, dim, 2),
    }
    targets = {
        'token_ids': torch.randint(0, vocab_size, (batch, seq)),
    }
    
    # CE objective
    ce = CrossEntropyObjective()
    result = ce(model_output, targets)
    assert result.loss.dim() == 0
    assert 'ce_loss' in result.metrics
    print("  ✓ CrossEntropyObjective")
    
    # Coherence objective
    coh = CoherenceObjective()
    result = coh(model_output, targets)
    assert result.loss.dim() == 0
    assert 'avg_coherence' in result.metrics
    print("  ✓ CoherenceObjective")
    
    # Energy objective
    energy = EnergyObjective()
    result = energy(model_output, targets)
    assert result.loss.dim() == 0
    assert 'avg_magnitude' in result.metrics
    print("  ✓ EnergyObjective")

    # Coupling objective
    coupling = CouplingObjective()
    context = {'coupling_loss': torch.tensor(0.5)}
    result = coupling(model_output, targets, context)
    assert result.loss.item() == 0.5
    assert 'coupling_loss' in result.metrics
    print("  ✓ CouplingObjective")

    print("\n✅ All Objective tests passed!")


def test_sampler():
    """Test Sampler"""
    print("\n" + "=" * 60)
    print("Testing Sampler...")
    print("=" * 60)
    
    from v4.sampler import AutoregressiveSampler
    
    batch, vocab_size = 2, 1000
    logits = torch.randn(batch, vocab_size)
    
    sampler = AutoregressiveSampler()
    
    # Basic sampling
    tokens, log_probs = sampler.sample(logits)
    assert tokens.shape == (batch, 1)
    assert log_probs.shape == (batch, 1)
    print("  ✓ AutoregressiveSampler basic")
    
    # With top-k
    tokens, _ = sampler.sample(logits, top_k=50)
    assert tokens.shape == (batch, 1)
    print("  ✓ AutoregressiveSampler top_k")
    
    # With top-p
    tokens, _ = sampler.sample(logits, top_p=0.9)
    assert tokens.shape == (batch, 1)
    print("  ✓ AutoregressiveSampler top_p")
    
    print("\n✅ All Sampler tests passed!")


def test_model():
    """Test full model"""
    print("\n" + "=" * 60)
    print("Testing Full Model...")
    print("=" * 60)
    
    from v4.model import create_model, QuantumPhaseFieldLLM
    from v4.core.config import get_default_config
    
    # Create tiny model for testing
    config = get_default_config('tiny')
    config.vocab_size = 1000
    config.dim = 32
    config.backbone.dim = 32
    config.backbone.state_dim = 64
    config.backbone.num_layers = 2
    config.memory.dim = 32
    config.memory.num_slots = 64
    for bank_cfg in config.banks.values():
        bank_cfg.dim = 32
    
    model = create_model(config=config)
    
    # Count parameters
    params = model.count_parameters()
    print(f"  Total parameters: {params['total']:,}")
    
    # Forward pass
    batch, seq = 2, 16
    input_ids = torch.randint(1, config.vocab_size, (batch, seq))
    
    output = model(input_ids)
    assert output.logits.shape == (batch, seq, config.vocab_size)
    assert output.phase_states.shape == (batch, seq, config.dim, 2)
    print("  ✓ Forward pass")
    
    # Generation
    prompt = torch.randint(1, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10)
    assert generated.shape[0] == 1
    assert generated.shape[1] == 15  # 5 prompt + 10 generated
    print("  ✓ Generation")
    
    # Backward pass
    loss = output.logits.mean()
    loss.backward()
    print("  ✓ Backward pass")
    
    print("\n✅ All Model tests passed!")


def test_byte_patching():
    """Test byte patching module"""
    print("\n" + "=" * 60)
    print("Testing Byte Patching...")
    print("=" * 60)
    
    from v4.core.byte_patching import BytePatcher, WithinPatchByteDecoder, BytePatchingModule
    
    batch, seq = 2, 17  # Non-multiple of patch_size to test padding
    dim = 64
    vocab_size = 259  # 256 bytes + PAD + BOS + EOS
    patch_size = 4
    
    # Test BytePatcher
    patcher = BytePatcher(vocab_size=vocab_size, dim=dim, patch_size=patch_size)
    
    input_ids = torch.randint(0, 256, (batch, seq))  # Random byte IDs
    patch_latents, info = patcher(input_ids)
    
    # Check shapes
    expected_patches = (seq + patch_size - 1) // patch_size  # Ceiling division
    assert patch_latents.shape == (batch, expected_patches, dim, 2), \
        f"Expected {(batch, expected_patches, dim, 2)}, got {patch_latents.shape}"
    assert info.original_len == seq
    assert info.patch_size == patch_size
    print(f"  ✓ BytePatcher: input {seq} bytes → {info.num_patches} patches")
    
    # Test WithinPatchByteDecoder
    decoder = WithinPatchByteDecoder(
        vocab_size=vocab_size, dim=dim, patch_size=patch_size, num_layers=2
    )
    
    # Simulate backbone output (same shape as patch_latents)
    patch_states = torch.randn(batch, info.num_patches, dim, 2)
    logits = decoder(patch_states, input_ids, info)
    
    assert logits.shape == (batch, seq, vocab_size), \
        f"Expected {(batch, seq, vocab_size)}, got {logits.shape}"
    print(f"  ✓ WithinPatchByteDecoder: patches → logits {logits.shape}")
    
    # Test BytePatchingModule (combined)
    module = BytePatchingModule(
        vocab_size=vocab_size, dim=dim, patch_size=patch_size, decoder_layers=2
    )
    
    latents, info = module.encode(input_ids)
    logits = module.decode(torch.randn_like(latents), input_ids, info)
    assert logits.shape == (batch, seq, vocab_size)
    print("  ✓ BytePatchingModule encode/decode")
    
    # Test gradient flow through the full module
    module.zero_grad()
    latents2, info2 = module.encode(input_ids)
    logits2 = module.decode(latents2, input_ids, info2)  # Use actual latents, not random
    logits2.mean().backward()
    assert module.patcher.position_weights.grad is not None
    print("  ✓ Gradient flow through byte patching")
    
    print("\n✅ All Byte Patching tests passed!")


def test_byte_patching_model():
    """Test full model with byte patching"""
    print("\n" + "=" * 60)
    print("Testing Model with Byte Patching...")
    print("=" * 60)
    
    from v4.model import create_model
    from v4.core.config import get_default_config, BytePatchingConfig
    
    # Create config for byte mode with patching
    config = get_default_config('tiny')
    config.vocab_size = 259  # Byte vocabulary
    config.dim = 32
    config.backbone.dim = 32
    config.backbone.state_dim = 64
    config.backbone.num_layers = 2
    config.memory.dim = 32
    config.memory.num_slots = 64
    for bank_cfg in config.banks.values():
        bank_cfg.dim = 32
    
    # Enable byte patching
    config.tokenizer.mode = 'byte'
    config.tokenizer.byte_patching = BytePatchingConfig(
        enabled=True,
        patch_size=4,
        decoder_layers=2,
    )
    
    model = create_model(config=config)
    assert model.use_byte_patching, "Byte patching should be enabled"
    print("  ✓ Model created with byte patching")
    
    # Forward pass
    batch, seq = 2, 20
    input_ids = torch.randint(0, 256, (batch, seq))  # Random bytes
    
    output = model(input_ids)
    
    # Logits should be [batch, seq, 259] - per-byte predictions
    assert output.logits.shape == (batch, seq, 259), \
        f"Expected logits {(batch, seq, 259)}, got {output.logits.shape}"
    print(f"  ✓ Forward pass: logits shape {output.logits.shape}")
    
    # Phase states should be [batch, num_patches, dim, 2]
    expected_patches = (seq + 3) // 4  # ceil(seq / 4)
    assert output.phase_states.shape == (batch, expected_patches, config.dim, 2), \
        f"Expected phase_states {(batch, expected_patches, config.dim, 2)}, got {output.phase_states.shape}"
    print(f"  ✓ Phase states shape: {output.phase_states.shape}")
    
    # byte_patch_info should be present
    assert output.byte_patch_info is not None
    assert output.byte_patch_info.original_len == seq
    print("  ✓ BytePatchInfo present")
    
    # Test loss computation
    import torch.nn.functional as F
    shift_logits = output.logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, 259),
        shift_targets.view(-1),
        ignore_index=256,  # PAD token
    )
    assert not torch.isnan(loss)
    print(f"  ✓ CE loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print("  ✓ Backward pass")
    
    print("\n✅ All Byte Patching Model tests passed!")


def test_registry():
    """Test registry system"""
    print("\n" + "=" * 60)
    print("Testing Registry...")
    print("=" * 60)
    
    from v4.core.registry import get_registry
    
    # Get the registry (modules were registered on import)
    registry = get_registry()
    
    # Check registrations
    banks_list = registry.list_banks()
    print(f"  Registered banks: {banks_list}")
    assert 'semantic' in banks_list, f"'semantic' not in {banks_list}"
    assert 'context' in banks_list
    print("  ✓ Banks registered")
    
    assert 'oscillatory_ssm' in registry.list_backbones()
    print("  ✓ Backbone registered")
    
    assert 'interference' in registry.list_couplers()
    print("  ✓ Coupler registered")
    
    assert 'phase_associative' in registry.list_memories()
    print("  ✓ Memory registered")
    
    assert 'ce' in registry.list_objectives()
    assert 'coherence' in registry.list_objectives()
    print("  ✓ Objectives registered")
    
    assert 'autoregressive' in registry.list_samplers()
    print("  ✓ Sampler registered")
    
    # Test factory creation
    bank = registry.create_bank('semantic', dim=64, num_concepts=32, num_layers=1)
    assert bank is not None
    print("  ✓ Factory creation works")
    
    print("\n✅ All Registry tests passed!")


def main():
    print("\n" + "=" * 60)
    print("v4 Quantum Phase-Field LLM - Test Suite")
    print("=" * 60)
    
    try:
        test_phase2d()
        test_banks()
        test_backbone()
        test_coupler()
        test_memory()
        test_objectives()
        test_sampler()
        test_byte_patching()
        test_registry()
        test_model()
        test_byte_patching_model()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
