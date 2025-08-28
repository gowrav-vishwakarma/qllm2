#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to run the quantum-inspired LLM training
"""

import os
import subprocess
import sys

def main():
    print("🚀 Starting Quantum-Inspired LLM Training")
    print("=" * 50)
    
    # Check if CUDA is available
    import torch
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please install PyTorch with CUDA support.")
        return
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    
    # Set environment variables for better performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Default training command
    cmd = [
        "python", "quantum_llm_train.py",
        "--mode", "train",
        "--model_dim", "512",
        "--num_layers", "8",
        "--num_heads", "8",
        "--phase_dim", "64",
        "--seq_length", "512",
        "--batch_size", "8",
        "--max_samples", "100000",
        "--epochs", "10",
        "--lr", "3e-4",
        "--energy_weight", "0.1",
        "--coherence_weight", "0.05",
        "--checkpoint_dir", "checkpoints_quantum",
        "--save_every", "500",
        "--log_every", "50",
        "--dataset", "wikitext2",
        "--use_checkpoint"
    ]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    print("=" * 50)
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()