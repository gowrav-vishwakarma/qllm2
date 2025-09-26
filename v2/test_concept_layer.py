#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Concept Layer Implementation
"""

import torch
import torch.nn as nn
from quantum_llm_model import HardwareOptimizedQuantumLLM
from concept_layer import ConceptLayer, ConceptLoss

def test_concept_layer():
    """Test the concept layer implementation"""
    print("🧪 Testing Concept Layer Implementation")
    print("=" * 60)
    
    # Test parameters
    vocab_size = 256
    concept_dim = 256
    num_concepts = 512
    batch_size = 4
    seq_len = 32
    
    # Create concept layer
    concept_layer = ConceptLayer(
        vocab_size=vocab_size,
        concept_dim=concept_dim,
        num_concepts=num_concepts,
        num_layers=2,
        dropout=0.1
    )
    
    print(f"✅ Concept layer created with {num_concepts} concepts")
    print(f"✅ Concept dimension: {concept_dim}")
    print(f"✅ Vocabulary size: {vocab_size}")
    
    # Test input
    token_embeddings = torch.randn(batch_size, seq_len, 512)  # Standard embedding size
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"✅ Test input shape: {token_embeddings.shape}")
    print(f"✅ Token IDs shape: {token_ids.shape}")
    
    # Test forward pass
    concept_repr, concept_logits = concept_layer(token_embeddings, token_ids)
    
    print(f"✅ Concept representation shape: {concept_repr.shape}")
    print(f"✅ Concept logits shape: {concept_logits.shape}")
    
    # Test concept analysis
    concept_analysis = concept_layer.get_concept_representation(token_ids)
    attention_weights = concept_layer.get_concept_attention_weights(token_ids)
    
    print(f"✅ Concept analysis shape: {concept_analysis.shape}")
    print(f"✅ Attention weights shape: {attention_weights.shape}")
    
    # Test concept loss
    concept_loss_fn = ConceptLoss(concept_dim=concept_dim, num_concepts=num_concepts)
    loss_dict = concept_loss_fn(concept_repr, token_ids)
    
    print(f"✅ Concept loss components:")
    for key, value in loss_dict.items():
        print(f"   {key}: {value.item():.6f}")
    
    print("\n✅ Concept layer test completed successfully!")

def test_integrated_model():
    """Test the concept layer integrated with the quantum model"""
    print("\n🧪 Testing Integrated Quantum Model with Concept Layer")
    print("=" * 60)
    
    # Model parameters
    vocab_size = 256
    dim = 768
    num_layers = 12
    num_heads = 12
    phase_dim = 128
    concept_dim = 256
    max_seq_len = 512
    
    # Create model
    model = HardwareOptimizedQuantumLLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        phase_dim=phase_dim,
        max_seq_len=max_seq_len,
        use_checkpoint=False,
        concept_dim=concept_dim
    )
    
    print(f"✅ Model created with concept layer")
    print(f"✅ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Concept layer parameters: {sum(p.numel() for p in model.concept_layer.parameters()):,}")
    
    # Test input
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"✅ Test input shape: {x.shape}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        logits = model(x)
        print(f"✅ Output logits shape: {logits.shape}")
        
        # Test concept analysis
        concept_analysis = model.get_concept_analysis(x)
        print(f"✅ Concept analysis keys: {list(concept_analysis.keys())}")
        print(f"✅ Concept representation shape: {concept_analysis['concept_representation'].shape}")
        print(f"✅ Attention weights shape: {concept_analysis['attention_weights'].shape}")
        
        # Test phase representation
        phase_repr = model.get_phase_representation(x)
        print(f"✅ Phase representation shape: {phase_repr.shape}")
        print(f"✅ Phase representation is complex: {phase_repr.is_complex()}")
    
    print("\n✅ Integrated model test completed successfully!")

def test_memory_usage():
    """Test memory usage of the concept layer"""
    print("\n🧪 Testing Memory Usage")
    print("=" * 60)
    
    # Get initial memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"✅ Initial VRAM: {initial_memory:.3f} GB")
    
    # Create model
    model = HardwareOptimizedQuantumLLM(
        vocab_size=256,
        dim=768,
        num_layers=12,
        num_heads=12,
        phase_dim=128,
        max_seq_len=512,
        use_checkpoint=False,
        concept_dim=256
    ).cuda()
    
    if torch.cuda.is_available():
        model_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"✅ Model VRAM: {model_memory:.3f} GB")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    x = torch.randint(0, 256, (batch_size, seq_len)).cuda()
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
        
        if torch.cuda.is_available():
            forward_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"✅ Forward pass VRAM: {forward_memory:.3f} GB")
            print(f"✅ Memory increase: {forward_memory - model_memory:.3f} GB")
    
    print("\n✅ Memory usage test completed!")

if __name__ == "__main__":
    # Run tests
    test_concept_layer()
    test_integrated_model()
    test_memory_usage()
    
    print("\n🎉 All concept layer tests completed successfully!")
    print("The concept layer is ready for training!")
