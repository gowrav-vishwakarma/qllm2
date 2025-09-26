#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Advanced Phase Coherence Implementation - Phase 2.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
from quantum_llm_model import AdvancedPhaseCoherence

def get_memory_usage():
    """Get current memory usage"""
    ram_gb = psutil.virtual_memory().used / (1024**3)
    try:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / (1024**3)
        else:
            vram_gb = 0
    except:
        vram_gb = 0
    return ram_gb, vram_gb

def test_advanced_phase_coherence():
    """Test the Advanced Phase Coherence implementation"""
    print("🧪 Testing Advanced Phase Coherence Implementation - Phase 2.2")
    print("=" * 70)
    
    # Configuration
    batch_size = 8
    seq_len = 128
    dim = 768
    phase_dim = 128
    vocab_size = 50257  # GPT-2 vocab size
    
    print(f"Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Embedding Dim: {dim}")
    print(f"  Phase Dim: {phase_dim}")
    print(f"  Vocab Size: {vocab_size}")
    print()
    
    # Initialize memory tracking
    start_ram, start_vram = get_memory_usage()
    start_time = time.time()
    
    # Create test data
    print("📊 Creating test data...")
    embeddings = torch.randn(batch_size, seq_len, dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda' if torch.cuda.is_available() else 'cpu')
    context_mask = torch.ones(batch_size, seq_len, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Advanced Phase Coherence layer
    print("🔧 Initializing Advanced Phase Coherence layer...")
    advanced_coherence = AdvancedPhaseCoherence(
        dim=dim,
        phase_dim=phase_dim,
        vocab_size=vocab_size,
        num_scales=5,
        num_semantic_clusters=64
    )
    
    if torch.cuda.is_available():
        advanced_coherence = advanced_coherence.cuda()
    
    # Test forward pass
    print("🚀 Testing forward pass...")
    coherence_start = time.time()
    
    with torch.no_grad():
        enhanced_embeddings = advanced_coherence(embeddings, token_ids, context_mask)
    
    coherence_time = time.time() - coherence_start
    
    # Test coherence metrics
    print("📈 Testing coherence metrics...")
    metrics_start = time.time()
    
    with torch.no_grad():
        coherence_metrics = advanced_coherence.get_coherence_metrics(embeddings, token_ids)
    
    metrics_time = time.time() - metrics_start
    
    # Get final memory usage
    end_ram, end_vram = get_memory_usage()
    total_time = time.time() - start_time
    
    # Results
    print("\n✅ Test Results:")
    print(f"  Enhanced Embeddings Shape: {enhanced_embeddings.shape}")
    print(f"  Expected Shape: {embeddings.shape}")
    print(f"  Shape Match: {enhanced_embeddings.shape == embeddings.shape}")
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Forward Pass Time: {coherence_time:.4f}s")
    print(f"  Metrics Calculation Time: {metrics_time:.4f}s")
    print(f"  Total Test Time: {total_time:.4f}s")
    
    print(f"\n💾 Memory Usage:")
    print(f"  RAM Used: {end_ram - start_ram:.2f}GB")
    print(f"  VRAM Used: {end_vram - start_vram:.2f}GB")
    
    print(f"\n📈 Coherence Metrics:")
    for metric_name, metric_value in coherence_metrics.items():
        if isinstance(metric_value, torch.Tensor):
            if metric_value.dim() > 0:
                print(f"  {metric_name}: {metric_value.shape} - Mean: {metric_value.mean().item():.4f}")
            else:
                print(f"  {metric_name}: {metric_value.item():.4f}")
        else:
            print(f"  {metric_name}: {metric_value}")
    
    # Test different coherence types
    print(f"\n🔍 Testing Individual Coherence Types:")
    
    # Test multi-scale coherence
    scales = torch.tensor([1, 2, 4, 8, 16], dtype=torch.float32)
    phases = torch.tanh(embeddings)
    
    from quantum_llm_model import compute_multi_scale_coherence
    multi_scale_result = compute_multi_scale_coherence(phases, scales)
    print(f"  Multi-scale Coherence Shape: {multi_scale_result.shape}")
    print(f"  Multi-scale Coherence Mean: {multi_scale_result.mean().item():.4f}")
    
    # Test semantic coherence
    from quantum_llm_model import compute_semantic_coherence
    # Use the same approach as in the class
    token_embeddings = advanced_coherence.token_similarity[token_ids]  # [batch, seq, vocab_size]
    token_emb_norm = F.normalize(token_embeddings, dim=-1)
    token_sim = torch.bmm(token_emb_norm, token_emb_norm.transpose(1, 2))  # [batch, seq, seq]
    semantic_result = compute_semantic_coherence(phases, token_sim)
    print(f"  Semantic Coherence Shape: {semantic_result.shape}")
    print(f"  Semantic Coherence Mean: {semantic_result.mean().item():.4f}")
    
    # Test context coherence
    context_result = advanced_coherence._compute_context_coherence(embeddings)
    print(f"  Context Coherence Shape: {context_result.shape}")
    print(f"  Context Coherence Mean: {context_result.mean().item():.4f}")
    
    # Test local coherence
    local_result = advanced_coherence._compute_local_coherence(embeddings)
    print(f"  Local Coherence Shape: {local_result.shape}")
    print(f"  Local Coherence Mean: {local_result.mean().item():.4f}")
    
    # Validation checks
    print(f"\n✅ Validation Checks:")
    
    # Check that output has same shape as input
    shape_valid = enhanced_embeddings.shape == embeddings.shape
    print(f"  Output Shape Valid: {shape_valid}")
    
    # Check that coherence metrics are reasonable
    metrics_valid = all(
        isinstance(v, torch.Tensor) and not torch.isnan(v).any() and not torch.isinf(v).any()
        for v in coherence_metrics.values()
    )
    print(f"  Coherence Metrics Valid: {metrics_valid}")
    
    # Check that enhanced embeddings are not identical to input
    not_identical = not torch.allclose(enhanced_embeddings, embeddings, atol=1e-6)
    print(f"  Enhanced Embeddings Modified: {not_identical}")
    
    # Check that coherence values are in reasonable range
    coherence_range_valid = torch.all(enhanced_embeddings >= -10) and torch.all(enhanced_embeddings <= 10)
    print(f"  Coherence Range Valid: {coherence_range_valid}")
    
    # Overall test result
    all_tests_passed = shape_valid and metrics_valid and not_identical and coherence_range_valid
    print(f"\n🎯 Overall Test Result: {'✅ PASSED' if all_tests_passed else '❌ FAILED'}")
    
    if all_tests_passed:
        print("\n🚀 Advanced Phase Coherence Implementation is working correctly!")
        print("   Ready for integration into the main model.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    return all_tests_passed

if __name__ == "__main__":
    try:
        success = test_advanced_phase_coherence()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
