#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for text generation with trained quantum-inspired LLM
"""

import subprocess
import sys
import os

def test_generation():
    print("🧪 Testing Quantum-Inspired LLM Generation")
    print("=" * 50)
    
    # Test prompts
    prompts = [
        "The quantum computer",
        "Artificial intelligence will",
        "In the year 2050",
        "The scientist discovered",
        "Once upon a time"
    ]
    
    checkpoint_path = "checkpoints_quantum/best_perplexity.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please run training first or specify a different checkpoint path.")
        return
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 Test {i}: '{prompt}'")
        print("-" * 40)
        
        cmd = [
            "python", "quantum_llm_train.py",
            "--mode", "generate",
            "--checkpoint", checkpoint_path,
            "--prompt", prompt,
            "--max_new_tokens", "150",
            "--temperature", "0.7",
            "--top_k", "50",
            "--top_p", "0.9",
            "--repetition_penalty", "1.1",
            "--min_p", "0.05"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    test_generation()