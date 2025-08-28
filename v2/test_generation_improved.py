#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved test script for the trained Quantum-Inspired LLM with better sampling
"""

import subprocess
import sys
import os

def test_generation_improved():
    print("🧪 Testing Trained Quantum-Inspired LLM Generation (Improved)")
    print("=" * 60)
    
    # Test prompts
    prompts = [
        "The quantum computer",
        "Artificial intelligence will",
        "In the year 2050",
        "The scientist discovered",
        "Once upon a time",
        "Machine learning models",
        "The future of technology",
        "Quantum mechanics reveals"
    ]
    
    checkpoint_path = "checkpoints_quantum/best_perplexity.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for file in os.listdir("checkpoints_quantum"):
            if file.endswith(".pt"):
                print(f"  - {file}")
        return
    
    print(f"✅ Using checkpoint: {checkpoint_path}")
    print()
    
    # Test different sampling strategies
    sampling_configs = [
        {
            "name": "Conservative (Low Temp)",
            "temperature": 0.3,
            "top_k": 20,
            "top_p": 0.8,
            "repetition_penalty": 1.1
        },
        {
            "name": "Balanced",
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        },
        {
            "name": "Creative (High Temp)",
            "temperature": 1.0,
            "top_k": 100,
            "top_p": 0.95,
            "repetition_penalty": 1.05
        }
    ]
    
    for config in sampling_configs:
        print(f"\n🎯 Testing {config['name']} Sampling Strategy")
        print("=" * 50)
        
        for i, prompt in enumerate(prompts[:3], 1):  # Test first 3 prompts for each config
            print(f"\n📝 Test {i}: '{prompt}'")
            print("-" * 30)
            
            cmd = [
                "python", "quantum_llm_train.py",
                "--mode", "generate",
                "--checkpoint", checkpoint_path,
                "--prompt", prompt,
                "--max_new_tokens", "100",  # Shorter for testing
                "--temperature", str(config["temperature"]),
                "--top_k", str(config["top_k"]),
                "--top_p", str(config["top_p"]),
                "--repetition_penalty", str(config["repetition_penalty"]),
                "--min_p", "0.05",
                # Use the actual trained model parameters
                "--model_dim", "512",
                "--num_layers", "8",
                "--num_heads", "8",
                "--phase_dim", "64",
                "--seq_length", "512"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # Extract just the generated text
                lines = result.stdout.split('\n')
                generated_text = None
                for line in lines:
                    if line.startswith("Generated text:"):
                        generated_text = line.replace("Generated text:", "").strip()
                        break
                
                if generated_text:
                    print(f"Generated: {generated_text}")
                else:
                    print("No generated text found")
            except subprocess.CalledProcessError as e:
                print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    test_generation_improved()
