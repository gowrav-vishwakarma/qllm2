#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Current Model with Improved Generation Parameters
Uses your existing trained model with better sampling settings
"""

import subprocess
import os

def test_with_better_params():
    """Test the current model with improved generation parameters"""
    
    # Check if best model exists
    if not os.path.exists("checkpoints/best_perplexity.pt"):
        print("❌ No best model found at checkpoints/best_perplexity.pt")
        return False
    
    print("🧪 Testing current model with improved generation parameters")
    print("=" * 60)
    
    # Test prompts with different styles
    test_prompts = [
        "The quantum algorithm",
        "In the year 2024, artificial intelligence",
        "The scientist discovered that",
        "Once upon a time, there was a",
        "The neural network processed the data and"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: '{prompt}'")
        print("-" * 40)
        
        cmd = [
            "python", "quantum_llm_train.py",
            "--mode", "generate",
            "--checkpoint", "checkpoints/best_perplexity.pt",
            "--prompt", prompt,
            "--max_new_tokens", "150",
            "--temperature", "0.7",      # Lower temperature for less randomness
            "--top_k", "50",             # Top-k sampling
            "--top_p", "0.9",            # Nucleus sampling
            "--repetition_penalty", "1.1",
            "--min_p", "0.05"            # Minimum probability threshold
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Extract the generated text from output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if line.startswith("Generated:"):
                    generated_text = line.replace("Generated:", "").strip()
                    print(f"Generated: {generated_text}")
                    break
        except subprocess.CalledProcessError as e:
            print(f"❌ Generation failed: {e}")
            print(f"Error output: {e.stderr}")

def test_with_different_temperatures():
    """Test the same prompt with different temperature settings"""
    
    if not os.path.exists("checkpoints/best_perplexity.pt"):
        print("❌ No best model found")
        return
    
    print("\n🌡️  Testing temperature variations")
    print("=" * 40)
    
    base_prompt = "The quantum computer"
    temperatures = [0.5, 0.7, 0.9, 1.1]
    
    for temp in temperatures:
        print(f"\n🔥 Temperature: {temp}")
        print("-" * 20)
        
        cmd = [
            "python", "quantum_llm_train.py",
            "--mode", "generate",
            "--checkpoint", "checkpoints/best_perplexity.pt",
            "--prompt", base_prompt,
            "--max_new_tokens", "100",
            "--temperature", str(temp),
            "--top_k", "50",
            "--top_p", "0.9",
            "--repetition_penalty", "1.1"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if line.startswith("Generated:"):
                    generated_text = line.replace("Generated:", "").strip()
                    print(f"Output: {generated_text}")
                    break
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed with temperature {temp}")

def main():
    print("🔬 Quantum LLM Model Testing")
    print("Testing your current model with improved parameters")
    
    # Test with better generation parameters
    test_with_better_params()
    
    # Test temperature variations
    test_with_different_temperatures()
    
    print("\n✅ Testing complete!")
    print("\n💡 Tips for better results:")
    print("   - Lower temperature (0.5-0.7) for more coherent text")
    print("   - Use top-k and top-p together for better sampling")
    print("   - Consider training on more data for better fluency")

if __name__ == "__main__":
    main()
