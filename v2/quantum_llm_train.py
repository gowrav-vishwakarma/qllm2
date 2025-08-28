#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for Quantum-Inspired LLM
Combines the model, trainer, and data loading into a complete training pipeline
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
from qllm_utils import device_str, save_checkpoint

def train(args):
    """Main training function"""
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
    
    # Create trainer
    trainer = EnergyBasedTrainer(
        model,
        learning_rate=args.lr,
        energy_weight=args.energy_weight,
        coherence_weight=args.coherence_weight,
        grad_clip=args.grad_clip
    )
    
    # Create data loaders
    train_loader, val_loader = build_loaders(
        args.dataset,
        args.seq_length,
        args.batch_size,
        args.max_samples,
        streaming=args.streaming,
        num_workers=args.num_workers
    )
    
    # Setup logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'logs'))
    
    # Training loop
    global_step = 0
    best_val_ppl = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        total_ce = 0
        total_energy = 0
        total_coherence = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Training step
            metrics = trainer.training_step(batch)
            
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
                
                print(f"Epoch {epoch} Step {global_step}: "
                      f"Loss {metrics['loss']:.4f} | "
                      f"CE {metrics['ce_loss']:.4f} | "
                      f"Energy {metrics['energy']:.4f} | "
                      f"Coherence {metrics['coherence']:.4f} | "
                      f"LR {metrics['lr']:.6f}")
            
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
        
        # End of epoch validation
        val_metrics = trainer.validate(val_loader, device)
        
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
        
        # Save best model
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
    
    writer.close()
    print("Training completed!")

def generate(args):
    """Text generation using trained model"""
    device = device_str()
    
    # Create model
    model = HardwareOptimizedQuantumLLM(
        vocab_size=256,
        dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        phase_dim=args.phase_dim,
        max_seq_len=args.seq_length,
        use_checkpoint=False
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Simple generation (you can enhance with sampling methods)
    from sampling_qllm import sample_next_token
    
    # Encode prompt
    prompt_bytes = args.prompt.encode('utf-8', errors='ignore')
    context = list(prompt_bytes)[:args.seq_length]
    if len(context) == 0:
        context = [ord(' ')]
    
    x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    out_ids = list(context)
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--energy_weight", type=float, default=0.1)
    parser.add_argument("--coherence_weight", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint_dir", default="checkpoints_quantum")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    
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