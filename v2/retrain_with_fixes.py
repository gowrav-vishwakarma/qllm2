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
    
    print("🔄 Running FAST POC Training")
    print("=" * 50)
    
    # Check if we can resume from existing checkpoint
    resume_checkpoint = "checkpoints_quantum_fixed/checkpoint_latest.pt"
    can_resume = os.path.exists(resume_checkpoint)
    
    if can_resume:
        print(f"✅ Found existing checkpoint: {resume_checkpoint}")
        print("   Training will resume from this checkpoint")
        print()
    
    # FAST POC training command for quick validation
    cmd = [
        "uv", "run", "quantum_llm_train.py",
        "--mode", "train",
        "--dataset", "wikitext2",  # Small dataset for quick training
        "--streaming", "False",
        "--max_samples", "1000",  # Reduced for faster training
        "--epochs", "10",  # 4 epochs for proper validation
        "--batch_size", "8",  # Small batch size for stability
        "--model_dim", "768",  # Keep model capacity for meaningful results
        "--num_layers", "12",  # Keep depth for quantum effects
        "--num_heads", "12",  # Keep attention heads
        "--phase_dim", "128",  # Keep phase representation
        "--seq_length", "512",  # Reduced for faster training
        "--lr", "5e-4",  # Higher learning rate for faster convergence
        "--energy_weight", "0.001",  # Restored quantum training with fixed energy calculation
        "--coherence_weight", "0.0005",  # Restored quantum training with fixed energy calculation
        "--grad_clip", "1.0",  # Keep gradient clipping
        "--warmup_steps", "100",  # Reduced warmup for faster start
        "--checkpoint_dir", "checkpoints_quantum_fixed",
        "--save_every", "100",  # More frequent saves for monitoring
        "--log_every", "50",  # Very frequent logging for monitoring
        "--num_workers", "2",  # Reduced for faster data loading
        "--val_max_chunks", "500",  # Smaller validation set
        "--use_checkpoint",  # Keep checkpointing
        "--gradient_accumulation_steps", "2"  # Reduced for faster updates
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
    print("🚀 QLLM FAST POC - Quick Validation")
    print("=" * 50)
    print()
    print("This will run a FAST POC to validate our quantum-inspired architecture:")
    print("✅ 4 epochs training for proper validation")
    print("✅ Small dataset (10K samples) to avoid overfitting")
    print("✅ Reduced sequence length (512) for faster training")
    print("✅ Higher learning rate (5e-4) for faster convergence")
    print("✅ Frequent logging (every 50 steps) for monitoring")
    print("✅ Keep model capacity (768 dim, 12 layers) for meaningful results")
    print("✅ Quantum training RESTORED with fixed energy calculations")
    print()
    
    # Ask for confirmation
    # response = input("Proceed with retraining? (y/N): ")
    # if response.lower() != 'y':
    #     print("❌ Retraining cancelled")
    #     return
    
    # Run retraining
    success = retrain_with_fixes()
    
    if success:
        print("\n🎉 Retraining completed successfully!")
        print("The model should now generate much better text.")
    else:
        print("\n❌ Retraining failed. Check the logs for details.")

if __name__ == "__main__":
    main()
