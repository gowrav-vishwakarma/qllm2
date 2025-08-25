#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Quantum LLM Training Script
Based on feedback for better text generation quality
"""

import os
import subprocess
import sys

def run_training_command():
    """Run training with improved parameters for better text generation"""
    
    # Improved parameters based on LLM feedback
    cmd = [
        "python", "quantum_llm_train.py",
        "--mode", "train",
        "--dataset", "wikitext2",    # Use wikitext2 instead of c4_en_small
        "--max_samples", "100000",   # 100k samples for wikitext2
        "--epochs", "20",            # Train longer (20 epochs)
        "--batch_size", "16",        # Smaller batch for larger model
        "--accumulate_steps", "4",   # Effective batch size = 16 * 4 = 64
        "--model_dim", "768",        # Larger model (768d vs 512d)
        "--num_layers", "12",        # More layers (12 vs 8)
        "--num_heads", "12",         # More heads (12 vs 8)
        "--seq_length", "256",       # Longer sequences for better context
        "--global_tokens", "16",     # More global tokens for long-range structure
        "--lora_rank", "32",         # Higher LoRA rank for better adaptation
        "--lora_alpha", "64.0",      # Higher alpha for LoRA
        "--lr", "5e-5",              # Slightly lower learning rate for stability
        "--weight_decay", "0.01",
        "--dropout", "0.1",          # Add some dropout for regularization
        "--phase_coh", "0.05",       # Reduced phase coherence weight
        "--checkpoint_dir", "checkpoints_improved",
        "--save_every", "2000",      # Save less frequently
        "--log_every", "200",        # Log more frequently
        "--seed", "42"
    ]
    
    print("🚀 Starting improved training with parameters:")
    print("   Dataset: wikitext2 (100k samples)")
    print("   Model: 768d, 12 layers, 12 heads")
    print("   Sequence length: 256")
    print("   Global tokens: 16")
    print("   LoRA: rank 32, alpha 64.0")
    print("   Training: 20 epochs with gradient accumulation")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False

def run_generation_command():
    """Run generation with improved sampling parameters"""
    
    cmd = [
        "python", "quantum_llm_train.py",
        "--mode", "generate",
        "--checkpoint", "checkpoints_improved/best_perplexity.pt",
        "--prompt", "The quantum computer processed the data with remarkable efficiency, revealing patterns that classical algorithms could never discover.",
        "--max_new_tokens", "300",
        "--temperature", "0.7",      # Lower temperature for less randomness
        "--top_k", "50",             # Top-k sampling
        "--top_p", "0.9",            # Nucleus sampling
        "--repetition_penalty", "1.1",
        "--min_p", "0.05"            # Minimum probability threshold
    ]
    
    print("🎯 Generating text with improved sampling parameters:")
    print("   Temperature: 0.7 (less random)")
    print("   Top-k: 50, Top-p: 0.9")
    print("   Repetition penalty: 1.1")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed with error: {e}")
        return False

def main():
    print("🔬 Quantum LLM Training Improvements")
    print("=" * 50)
    
    # Check if we want to train or just generate
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # Just generate with existing model
        if os.path.exists("checkpoints_improved/best_perplexity.pt"):
            run_generation_command()
        else:
            print("❌ No improved model found. Run training first.")
            print("   Usage: python train_improved.py")
    else:
        # Train the improved model
        success = run_training_command()
        if success:
            print("\n🎉 Training complete! Now generating sample text...")
            run_generation_command()

if __name__ == "__main__":
    main()
