#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for Quantum-Inspired LLM with epoch-based scheduler
"""

import os
import math
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from quantum_llm_model import HardwareOptimizedQuantumLLM
from energy_trainer import EnergyBasedTrainer
from datasets_qllm import build_loaders
from qllm_utils import device_str, save_checkpoint, get_memory_usage

def train(args):
    """Main training function with memory optimizations"""
    # Setup device
    device = device_str()
    print(f"Using device: {device}")
    
    # Create model
    model = HardwareOptimizedQuantumLLM(
        vocab_size=256,  # Byte-level vocabulary
        dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        phase_dim=args.phase_dim,
        max_seq_len=args.seq_length,
        use_checkpoint=args.use_checkpoint
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders with streaming and validation limits
    train_loader, val_loader = build_loaders(
        args.dataset,
        args.seq_length,
        args.batch_size,
        args.max_samples,
        streaming=args.streaming,
        num_workers=args.num_workers,
        val_max_chunks=args.val_max_chunks
    )
    
    # Create trainer with flexible scheduler
    trainer = EnergyBasedTrainer(
        model,
        learning_rate=args.lr,
        energy_weight=args.energy_weight,
        coherence_weight=args.coherence_weight,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps if args.max_steps else 10000
    )
    
    # Print validation dataset size
    print(f"📊 Validation dataset size: {len(val_loader.dataset)} chunks")
    print(f"📊 Validation batches: {len(val_loader)}")
    
    # Setup logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'logs'))
    
    # Training loop
    global_step = 0
    best_val_ppl = float('inf')
    patience = 0
    max_patience = 5  # Early stopping patience
    
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        total_ce = 0
        total_energy = 0
        total_coherence = 0
        num_batches = 0
        
        # Print memory usage before epoch
        mem_usage = get_memory_usage()
        print(f"Memory usage before epoch {epoch}: {mem_usage}")
        
        for batch_idx, batch in enumerate(train_loader):
            # The batch is now a tuple of (inputs, targets) thanks to collate_fn
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Skip incomplete batches
            if inputs.size(0) != args.batch_size:
                continue
            
            # Training step
            metrics = trainer.training_step(inputs, targets)
            
            # Accumulate metrics
            total_loss += metrics['loss']
            total_ce += metrics['ce_loss']
            total_energy += metrics['energy']
            total_coherence += metrics['coherence']
            num_batches += 1
            
            # Log to tensorboard
            if global_step % args.log_every == 0:
                writer.add_scalar('Train/Loss', metrics['loss'], global_step)
                writer.add_scalar('Train/CE', metrics['ce_loss'], global_step)
                writer.add_scalar('Train/Energy', metrics['energy'], global_step)
                writer.add_scalar('Train/Coherence', metrics['coherence'], global_step)
                writer.add_scalar('Train/LR', metrics['lr'], global_step)
                
                # Print memory usage
                mem_usage = get_memory_usage()
                print(f"Epoch {epoch} Step {global_step}: "
                      f"Loss {metrics['loss']:.4f} | "
                      f"CE {metrics['ce_loss']:.4f} | "
                      f"Energy {metrics['energy']:.4f} | "
                      f"Coherence {metrics['coherence']:.4f} | "
                      f"LR {metrics['lr']:.6f} | "
                      f"RAM {mem_usage.get('allocated', 0):.2f}GB")
            
            # Save checkpoint
            if global_step % args.save_every == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_step_{global_step}.pt')
                save_checkpoint({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                    'args': vars(args)
                }, checkpoint_path)
                
                # Also save as latest
                latest_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pt')
                save_checkpoint({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                    'args': vars(args)
                }, latest_path)
            
            global_step += 1
            
            # Break if we've processed enough samples
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        
        # End of epoch validation
        print(f"\n🔄 Starting validation for epoch {epoch}...")
        val_start_time = time.time()
        val_metrics = trainer.validate(val_loader, device)
        val_time = time.time() - val_start_time
        print(f"✅ Validation completed for epoch {epoch} in {val_time:.2f}s")
        
        # Log validation metrics
        writer.add_scalar('Val/Loss', val_metrics['val_loss'], global_step)
        writer.add_scalar('Val/CE', val_metrics['val_ce'], global_step)
        writer.add_scalar('Val/Energy', val_metrics['val_energy'], global_step)
        writer.add_scalar('Val/Coherence', val_metrics['val_coherence'], global_step)
        writer.add_scalar('Val/PPL', val_metrics['val_ppl'], global_step)
        
        print(f"Epoch {epoch} completed in {time.time() - epoch_start:.2f}s")
        print(f"Validation: Loss {val_metrics['val_loss']:.4f} | "
              f"CE {val_metrics['val_ce']:.4f} | "
              f"Energy {val_metrics['val_energy']:.4f} | "
              f"Coherence {val_metrics['val_coherence']:.4f} | "
              f"PPL {val_metrics['val_ppl']:.2f}")
        
        # Early stopping check
        if val_metrics['val_ppl'] < best_val_ppl:
            best_val_ppl = val_metrics['val_ppl']
            best_path = os.path.join(args.checkpoint_dir, 'best_perplexity.pt')
            save_checkpoint({
                'model_state_dict': model.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
                'val_ppl': val_metrics['val_ppl'],
                'args': vars(args)
            }, best_path)
            print(f"New best model saved with perplexity {best_val_ppl:.2f}")
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Clear cache between epochs
        torch.cuda.empty_cache()
    
    writer.close()
    print("Training completed!")

def generate(args):
    """Text generation using trained model"""
    device = device_str()
    
    # Load checkpoint first to get the model parameters
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract model parameters from checkpoint
    if 'args' in checkpoint:
        # Use parameters from checkpoint
        model_dim = checkpoint['args'].get('model_dim', 384)
        num_layers = checkpoint['args'].get('num_layers', 6)
        num_heads = checkpoint['args'].get('num_heads', 6)
        phase_dim = checkpoint['args'].get('phase_dim', 48)
        seq_length = checkpoint['args'].get('seq_length', 256)
    else:
        # Fallback to command line args or defaults
        model_dim = getattr(args, 'model_dim', 384)
        num_layers = getattr(args, 'num_layers', 6)
        num_heads = getattr(args, 'num_heads', 6)
        phase_dim = getattr(args, 'phase_dim', 48)
        seq_length = getattr(args, 'seq_length', 256)
    
    print(f" Loading model with parameters: dim={model_dim}, layers={num_layers}, heads={num_heads}, phase_dim={phase_dim}")
    
    # Create model with correct parameters
    model = HardwareOptimizedQuantumLLM(
        vocab_size=256,
        dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        phase_dim=phase_dim,
        max_seq_len=seq_length,
        use_checkpoint=False
    ).to(device)
    
    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Simple generation (you can enhance with sampling methods)
    from sampling_qllm import sample_next_token
    
    # Encode prompt
    prompt_bytes = args.prompt.encode('utf-8', errors='ignore')
    context = list(prompt_bytes)[:seq_length]
    if len(context) == 0:
        context = [ord(' ')]
    
    x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    out_ids = list(context)
    
    print(f"🎯 Generating {args.max_new_tokens} tokens for prompt: '{args.prompt}'")
    
    # Generate tokens
    with torch.no_grad():
        for i in range(args.max_new_tokens):
            x_in = x[:, -model.max_seq_len:]
            logits = model(x_in)
            next_logits = logits[:, -1, :]
            next_id = sample_next_token(
                next_logits,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                recent_ids=[out_ids[-64:]],
                min_p=args.min_p
            ).item()
            out_ids.append(next_id)
            x = torch.tensor(out_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # Print progress every 50 tokens
            if (i + 1) % 50 == 0:
                print(f"   Generated {i + 1}/{args.max_new_tokens} tokens...")
    
    # Decode and print
    generated_text = bytes(out_ids).decode('utf-8', errors='ignore')
    print(f"\nGenerated text:\n{generated_text}")

def main():
    parser = argparse.ArgumentParser(description="Quantum-Inspired LLM Training")
    parser.add_argument("--mode", required=True, choices=["train", "generate"])
    
    # Model parameters
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--phase_dim", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--use_checkpoint", action="store_true")
    
    # Training parameters
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--energy_weight", type=float, default=0.1)
    parser.add_argument("--coherence_weight", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--checkpoint_dir", default="checkpoints_quantum")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_max_chunks", type=int, default=1000)  # Limit validation chunks
    
    # Generation parameters
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="The quantum computer")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--min_p", type=float, default=0.05)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    else:
        generate(args)

if __name__ == "__main__":
    main()