#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Generation Script with Better Parameters
Generate text using your trained model with improved settings
"""

import subprocess
import sys

def generate_text(prompt, temperature=0.7, max_tokens=200):
    """Generate text with improved parameters"""
    
    cmd = [
        "python", "quantum_llm_train.py",
        "--mode", "generate",
        "--checkpoint", "checkpoints_improved/best_perplexity.pt",
        "--prompt", prompt,
        "--max_new_tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--top_k", "50",
        "--top_p", "0.9",
        "--repetition_penalty", "1.1",
        "--min_p", "0.05"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Extract generated text
        for line in result.stdout.split('\n'):
            if line.startswith("Generated:"):
                return line.replace("Generated:", "").strip()
        return "No output found"
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_better.py 'Your prompt here' [temperature] [max_tokens]")
        print("Example: python generate_better.py 'The quantum computer' 0.7 150")
        sys.exit(1)
    
    prompt = sys.argv[1]
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    
    print(f"🎯 Generating with prompt: '{prompt}'")
    print(f"🌡️  Temperature: {temperature}")
    print(f"📏 Max tokens: {max_tokens}")
    print("-" * 50)
    
    result = generate_text(prompt, temperature, max_tokens)
    print(f"Generated: {result}")

if __name__ == "__main__":
    main()
