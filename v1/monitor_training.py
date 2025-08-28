#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor Large-Scale Quantum LLM Training
Real-time monitoring for the large-scale model training
"""

import os
import time
import json
import torch
from pathlib import Path

def get_gpu_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        try:
            # Get more accurate GPU memory info
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            
            # Also try to get memory from nvidia-smi if available
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        used_mb, total_mb = map(int, lines[0].split(', '))
                        used_gb = used_mb / 1024
                        total_gb = total_mb / 1024
                        # Use nvidia-smi values if they seem more reasonable
                        if used_gb > 0:
                            allocated = used_gb
                            total = total_gb
                            free = total - used_gb
            except:
                pass  # Fall back to torch values
                
            return {
                'memory_allocated': allocated,
                'memory_reserved': reserved,
                'memory_free': free,
                'memory_total': total
            }
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return None
    return None

def get_latest_checkpoint(checkpoint_dir):
    """Get the latest checkpoint file"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Look for the most recent checkpoint
    checkpoints = list(checkpoint_dir.glob("model_step*.pt"))
    if not checkpoints:
        return None
    
    return max(checkpoints, key=lambda x: x.stat().st_mtime)

def get_model_args(checkpoint_dir):
    """Load model arguments from JSON"""
    args_path = Path(checkpoint_dir) / "model_args.json"
    if args_path.exists():
        with open(args_path, 'r') as f:
            return json.load(f)
    return None

def monitor_training(checkpoint_dir="checkpoints_large_scale"):
    """Monitor training progress in real-time"""
    
    print("🔍 MONITORING LARGE-SCALE QUANTUM LLM TRAINING")
    print("=" * 60)
    
    # Load model configuration
    model_args = get_model_args(checkpoint_dir)
    if model_args:
        print(f"📋 Model Configuration:")
        print(f"   Model: {model_args.get('model_dim', 'N/A')}d, {model_args.get('num_layers', 'N/A')} layers")
        print(f"   Sequence: {model_args.get('seq_length', 'N/A')} tokens")
        print(f"   Attention: {model_args.get('attention_type', 'N/A')}")
        print(f"   Dataset: {model_args.get('dataset', 'N/A')}")
        print()
    
    last_checkpoint = None
    start_time = time.time()
    
    while True:
        try:
            # Clear screen (optional)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("🔍 LARGE-SCALE QUANTUM LLM TRAINING MONITOR")
            print("=" * 60)
            print(f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🕐 Elapsed: {time.time() - start_time:.0f}s")
            print()
            
            # GPU Memory
            gpu_info = get_gpu_info()
            if gpu_info:
                print("💻 GPU Memory Usage:")
                print(f"   Allocated: {gpu_info['memory_allocated']:.2f} GB")
                print(f"   Reserved:  {gpu_info['memory_reserved']:.2f} GB")
                print(f"   Free:      {gpu_info['memory_free']:.2f} GB")
                print(f"   Total:     {gpu_info['memory_total']:.2f} GB")
                print(f"   Utilization: {gpu_info['memory_allocated']/gpu_info['memory_total']*100:.1f}%")
                print()
            
            # Checkpoint status
            current_checkpoint = get_latest_checkpoint(checkpoint_dir)
            if current_checkpoint:
                if current_checkpoint != last_checkpoint:
                    print(f"✅ New checkpoint found: {current_checkpoint.name}")
                    last_checkpoint = current_checkpoint
                
                # Get checkpoint info
                stat = current_checkpoint.stat()
                print(f"📁 Latest checkpoint: {current_checkpoint.name}")
                print(f"   Size: {stat.st_size / 1024**2:.1f} MB")
                print(f"   Modified: {time.strftime('%H:%M:%S', time.localtime(stat.st_mtime))}")
                print()
            
            # Check for best perplexity
            best_ppl_path = Path(checkpoint_dir) / "best_perplexity.pt"
            if best_ppl_path.exists():
                stat = best_ppl_path.stat()
                print(f"🏆 Best model: {time.strftime('%H:%M:%S', time.localtime(stat.st_mtime))}")
                print()
            
            # Check for training log
            log_path = Path(checkpoint_dir) / "training.log"
            if log_path.exists():
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print("📊 Recent Training Log:")
                        for line in lines[-5:]:  # Last 5 lines
                            print(f"   {line.strip()}")
                        print()
            
            print("Press Ctrl+C to stop monitoring...")
            time.sleep(5)  # Update every 5 seconds
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    import sys
    
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_large_scale"
    monitor_training(checkpoint_dir)
