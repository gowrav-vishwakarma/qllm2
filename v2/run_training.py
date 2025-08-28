#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-efficient script to run the quantum-inspired LLM training
"""

import os
import subprocess
import sys

def main():
    print("🚀 Starting Memory-Efficient Quantum-Inspired LLM Training")
    print("=" * 60)
    
    # Check if CUDA is available
    import torch
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please install PyTorch with CUDA support.")
        return
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    
    # Set environment variables for better performance and memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only one GPU
    os.environ["PYTORCH_TRANSFORMERS_CACHE"] = "./cache"  # Cache directory
    
    # Create cache directory
    os.makedirs("./cache", exist_ok=True)
    
    # Memory-efficient training command with reduced parameters
    cmd = [
        "python", "quantum_llm_train.py",
        "--mode", "train",
        "--model_dim", "512",      # Reduced for memory efficiency
        "--num_layers", "8",       # Reduced for memory efficiency
        "--num_heads", "8",        # Reduced for memory efficiency
        "--phase_dim", "64",       # Reduced for memory efficiency
        "--seq_length", "512",     # Reduced for memory efficiency
        "--batch_size", "4",       # Reduced for memory efficiency
        "--max_samples", "50000",  # Reduced for memory efficiency
        "--epochs", "10",          # Train for 10 epochs for better quality
        # "--max_steps", "5000",    # Removed to allow full epoch training
        "--lr", "3e-4",            # Back to original learning rate
        "--energy_weight", "0.01", # Reduced from 0.02
        "--coherence_weight", "0.005", # Reduced from 0.01
        "--grad_clip", "1.0",
        "--warmup_steps", "500",   # Reduced for faster training
        "--checkpoint_dir", "checkpoints_quantum",
        "--save_every", "500",     # Reduced for more frequent saves
        "--log_every", "500",       # More frequent logging to see progress
        "--dataset", "wikitext2",
        "--use_checkpoint",
        "--streaming", "False",    # Disable streaming for better performance
        "--num_workers", "2",      # Reduced for memory efficiency
        "--val_max_chunks", "1000" # Reduced for memory efficiency
    ]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Print initial memory usage
    import psutil
    import torch
    memory = psutil.virtual_memory()
    gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    print(f"💾 Initial RAM usage: {memory.used / (1024**3):.2f}GB / {memory.total / (1024**3):.2f}GB")
    print(f"💾 Initial VRAM usage: {gpu_mem:.2f}GB / {gpu_total:.2f}GB")
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()