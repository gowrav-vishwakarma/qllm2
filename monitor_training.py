#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Monitor Script
Monitor the progress of the improved model training
"""

import os
import time
import subprocess
from datetime import datetime

def check_training_status():
    """Check if training is still running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return 'quantum_llm_train.py' in result.stdout
    except:
        return False

def get_latest_checkpoint():
    """Get the latest checkpoint file"""
    checkpoint_dir = "checkpoints_improved"
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return checkpoints[-1]

def get_file_size_mb(filepath):
    """Get file size in MB"""
    if os.path.exists(filepath):
        return round(os.path.getsize(filepath) / (1024 * 1024), 1)
    return 0

def monitor_training():
    """Monitor training progress"""
    print("🔬 Quantum LLM Training Monitor")
    print("=" * 50)
    
    start_time = time.time()
    last_checkpoint = None
    
    while True:
        # Check if training is still running
        if not check_training_status():
            print("❌ Training process not found. Training may have completed or failed.")
            break
        
        # Get current checkpoint
        current_checkpoint = get_latest_checkpoint()
        
        # Clear screen and show status
        os.system('clear')
        print("🔬 Quantum LLM Training Monitor")
        print("=" * 50)
        
        # Training time
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"⏱️  Training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Current checkpoint
        if current_checkpoint:
            checkpoint_path = os.path.join("checkpoints_improved", current_checkpoint)
            size_mb = get_file_size_mb(checkpoint_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
            
            print(f"💾 Latest checkpoint: {current_checkpoint}")
            print(f"📏 Size: {size_mb} MB")
            print(f"🕒 Modified: {mod_time.strftime('%H:%M:%S')}")
            
            # Check if new checkpoint was created
            if current_checkpoint != last_checkpoint:
                print(f"✅ New checkpoint created: {current_checkpoint}")
                last_checkpoint = current_checkpoint
        else:
            print("⏳ Waiting for first checkpoint...")
        
        # Model configuration
        print(f"\n📋 Model Configuration:")
        print(f"   • Model dimension: 768")
        print(f"   • Layers: 12")
        print(f"   • Heads: 12")
        print(f"   • Sequence length: 256")
        print(f"   • Global tokens: 16")
        print(f"   • LoRA rank: 32")
        
        # GPU usage
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                if len(gpu_info) >= 3:
                    gpu_util = gpu_info[0]
                    mem_used = gpu_info[1]
                    mem_total = gpu_info[2]
                    print(f"\n🖥️  GPU Usage:")
                    print(f"   • Utilization: {gpu_util}%")
                    print(f"   • Memory: {mem_used}/{mem_total} MB")
        except:
            pass
        
        print(f"\n🔄 Monitoring... (Press Ctrl+C to stop)")
        
        # Wait before next check
        time.sleep(10)

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")
