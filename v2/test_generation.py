#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the trained Quantum-Inspired LLM
"""

import subprocess
import sys
import os

def test_generation():
    print("🧪 Testing Trained Quantum-Inspired LLM Generation")
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
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 Test {i}: '{prompt}'")
        print("-" * 50)
        
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
            "--min_p", "0.05",
            # Add the model parameters that were used during training
            "--model_dim", "384",
            "--num_layers", "6",
            "--num_heads", "6",
            "--phase_dim", "48",
            "--seq_length", "256"
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
                print(generated_text)
            else:
                print("No generated text found")
                print("Full output:")
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Generation failed: {e}")
            print("Error output:")
            print(e.stderr)

if __name__ == "__main__":
    test_generation()