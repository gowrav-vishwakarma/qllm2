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
    
    # Improved training command with adjusted parameters
    cmd = [
        "python", "quantum_llm_train.py",
        "--mode", "train",
        "--model_dim", "384",  # Reduced model dimension
        "--num_layers", "6",   # Reduced layers
        "--num_heads", "6",    # Reduced heads
        "--phase_dim", "48",   # Reduced phase dimension
        "--seq_length", "256", # Reduced sequence length
        "--batch_size", "2",   # Reduced batch size
        "--max_samples", "30000",  # Reduced samples
        "--epochs", "10",
        "--max_steps", "2000",  # Limit total steps
        "--lr", "3e-4",
        "--energy_weight", "0.02",  # Further reduced energy weight
        "--coherence_weight", "0.01",  # Further reduced coherence weight
        "--warmup_steps", "200",  # Warmup steps for scheduler
        "--checkpoint_dir", "checkpoints_quantum",
        "--save_every", "500",
        "--log_every", "50",
        "--dataset", "wikitext2",
        "--use_checkpoint",
        "--streaming",  # Enable streaming
        "--num_workers", "1",  # Single worker to reduce memory
        "--val_max_chunks", "1000"  # Limit validation to 1000 chunks
    ]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Print initial memory usage
    import psutil
    memory = psutil.virtual_memory()
    print(f"💾 Initial RAM usage: {memory.used / (1024**3):.2f}GB / {memory.total / (1024**3):.2f}GB")
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()