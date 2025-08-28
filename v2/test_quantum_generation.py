#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for quantum-aware generation
"""

import torch
import argparse
from quantum_llm_model import HardwareOptimizedQuantumLLM
from quantum_llm_train import quantum_aware_generate

def test_quantum_generation():
    """Test quantum-aware generation with current model"""
    
    # Model parameters (matching the trained model)
    model_dim = 512
    num_layers = 8
    num_heads = 8
    phase_dim = 64
    seq_length = 512
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = HardwareOptimizedQuantumLLM(
        vocab_size=256,
        dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        phase_dim=phase_dim,
        max_seq_len=seq_length,
        use_checkpoint=False
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test prompts
    test_prompts = [
        "The quantum computer",
        "Artificial intelligence will",
        "In the year 2050",
        "The scientist discovered",
        "Once upon a time"
    ]
    
    print("\n🧪 Testing Quantum-Aware Generation")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: '{prompt}'")
        print("-" * 30)
        
        try:
            # Generate with quantum-aware sampling
            generated_text = quantum_aware_generate(
                model=model,
                prompt=prompt,
                tokenizer=None,
                max_new_tokens=50,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,  # Slightly higher to reduce repetition
                min_p=0.05
            )
            
            print(f"Generated: {generated_text}")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    print("\n✅ Quantum-aware generation test completed!")

if __name__ == "__main__":
    test_quantum_generation()
