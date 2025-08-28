#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain the model with the critical fixes for generation quality
"""

import os
import sys
import subprocess

def retrain_with_fixes():
    """Retrain the model with improved loss weights and semantic components"""
    
    print("🔄 Retraining QLLM with Critical Fixes")
    print("=" * 50)
    
    # Check if we can resume from existing checkpoint
    resume_checkpoint = "checkpoints_quantum_fixed/checkpoint_latest.pt"
    can_resume = os.path.exists(resume_checkpoint)
    
    if can_resume:
        print(f"✅ Found existing checkpoint: {resume_checkpoint}")
        print("   Training will resume from this checkpoint")
        print()
    
    # Training configuration with fixes
    cmd = [
        "uv", "run", "quantum_llm_train.py",
        "--mode", "train",
        "--dataset", "wikitext2",  # Start with smaller dataset for testing
        "--streaming", "False",
        "--max_samples", "20000",  # Increased for better training
        "--epochs", "5",  # Fewer epochs for testing
        "--batch_size", "16",  # Reduced from 32 to 16 for stability
        "--model_dim", "512",
        "--num_layers", "8",
        "--num_heads", "8",
        "--phase_dim", "64",
        "--seq_length", "512",
        "--lr", "1e-4",  # Reduced learning rate for stability
        "--energy_weight", "0.0005",  # Further reduced for stability
        "--coherence_weight", "0.0002",  # Further reduced for stability
        "--grad_clip", "0.5",  # Reduced gradient clipping
        "--warmup_steps", "500",  # Increased warmup
        "--checkpoint_dir", "checkpoints_quantum_fixed",
        "--save_every", "200",  # More frequent saves for monitoring
        "--log_every", "25",  # More frequent logging for monitoring
        "--num_workers", "4",  # Increased for faster data loading
        "--val_max_chunks", "1000",  # Increased validation set
        "--no_amp"  # Disable AMP for stability during testing
    ]
    
    print("Training command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Training completed successfully!")
        
        # Test generation with the new model
        print("\n🧪 Testing generation with fixed model...")
        test_generation()
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   - Check if NaN loss occurred (reduce batch_size further)")
        print("   - Check if out of memory (reduce batch_size)")
        print("   - Check if learning rate too high (reduce lr)")
        return False
    
    return True

def test_generation():
    """Test generation with the newly trained model"""
    
    checkpoint_path = "checkpoints_quantum_fixed/best_perplexity.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Test prompts
    test_prompts = [
        "The quantum computer",
        "Artificial intelligence will",
        "In the year 2050",
        "The scientist discovered",
        "Once upon a time"
    ]
    
    print(f"✅ Using checkpoint: {checkpoint_path}")
    print()
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: '{prompt}'")
        print("-" * 50)
        
        cmd = [
            "uv", "run", "quantum_llm_train.py",
            "--mode", "generate",
            "--checkpoint", checkpoint_path,
            "--prompt", prompt,
            "--max_new_tokens", "100",
            "--temperature", "0.7",
            "--top_k", "50",
            "--top_p", "0.9",
            "--repetition_penalty", "1.2",
            "--min_p", "0.05",
            "--model_dim", "512",
            "--num_layers", "8",
            "--num_heads", "8",
            "--phase_dim", "64",
            "--seq_length", "512"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Extract generated text
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

def main():
    """Main function"""
    print("🚀 QLLM Retraining with Critical Fixes")
    print("=" * 50)
    print()
    print("This will retrain the model with:")
    print("✅ Reduced quantum loss weights (energy: 0.001, coherence: 0.0005)")
    print("✅ Added semantic loss components")
    print("✅ Quantum-aware generation")
    print("✅ Better sampling strategies")
    print("✅ Optimized for RTX 4090 (batch_size: 16, stable training)")
    print("✅ Better VRAM utilization (target: 4-8GB usage)")
    print("✅ Stability-focused configuration (reduced LR, loss weights)")
    print()
    
    # Ask for confirmation
    response = input("Proceed with retraining? (y/N): ")
    if response.lower() != 'y':
        print("❌ Retraining cancelled")
        return
    
    # Run retraining
    success = retrain_with_fixes()
    
    if success:
        print("\n🎉 Retraining completed successfully!")
        print("The model should now generate much better text.")
    else:
        print("\n❌ Retraining failed. Check the logs for details.")

if __name__ == "__main__":
    main()
