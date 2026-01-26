#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Training with Real Data

Usage:
    # Quick test with WikiText-2
    python train_real.py --dataset wikitext2 --size small --epochs 5
    
    # TinyStories (good for small models)
    python train_real.py --dataset tinystories --size small --epochs 10
    
    # Full training
    python train_real.py --dataset tinystories --size medium --epochs 20 --batch_size 4
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from v4.model import create_model, QuantumPhaseFieldLLM
from v4.core.config import V4Config, get_default_config
from v4.core.registry import get_registry
from v4.data import get_wikitext2, get_tinystories, create_dataloaders, get_tokenizer


class RealDataTrainer:
    """Trainer for real dataset training with speed optimizations"""
    
    def __init__(
        self,
        model: QuantumPhaseFieldLLM,
        config: V4Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Apply torch.compile if enabled
        if config.training.compile_model and hasattr(torch, 'compile'):
            print(f"🔧 Compiling model with mode='{config.training.compile_mode}'...")
            try:
                self.model = torch.compile(
                    self.model, 
                    mode=config.training.compile_mode,
                    fullgraph=False,  # Allow graph breaks for flexibility
                )
                print("   ✅ Model compiled successfully")
            except Exception as e:
                print(f"   ⚠️ Compilation failed: {e}")
                print("   Continuing without compilation...")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.training.max_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
        )
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if config.training.mixed_precision and torch.cuda.is_available() else None
        
        # Objectives
        registry = get_registry()
        self.objectives = []
        for obj_cfg in config.objectives:
            obj = registry.create_objective(
                obj_cfg.type,
                weight=obj_cfg.weight,
                **obj_cfg.params
            )
            self.objectives.append(obj)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        num_batches = 0
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            
            # Forward
            with autocast('cuda', enabled=self.scaler is not None):
                output = self.model(input_ids)
                
                # Compute losses
                total_batch_loss = torch.tensor(0.0, device=self.device)
                
                model_output = {
                    'logits': output.logits,
                    'phase_states': output.phase_states,
                }
                targets = {'token_ids': input_ids}
                context = {'coupling_loss': output.coupling_loss}
                
                batch_ce = 0.0
                for objective in self.objectives:
                    result = objective(model_output, targets, context)
                    total_batch_loss = total_batch_loss + result.loss * objective.weight
                    
                    if objective.name == 'ce':
                        batch_ce = result.loss.item()
            
            # Backward
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                self.optimizer.step()

            # Scheduler should be stepped AFTER optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += total_batch_loss.item()
            total_ce_loss += batch_ce
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.config.training.log_every == 0:
                avg_loss = total_loss / num_batches
                avg_ce = total_ce_loss / num_batches
                ppl = torch.exp(torch.tensor(avg_ce)).item()
                lr = self.scheduler.get_last_lr()[0]
                
                elapsed = time.time() - epoch_start
                samples_per_sec = (batch_idx + 1) * self.config.training.batch_size / elapsed
                
                print(f"  [{batch_idx+1:4d}/{len(self.train_loader)}] "
                      f"Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | PPL: {ppl:.2f} | "
                      f"LR: {lr:.2e} | {samples_per_sec:.1f} samples/s")
        
        avg_loss = total_loss / num_batches
        avg_ce = total_ce_loss / num_batches
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce,
            'perplexity': torch.exp(torch.tensor(avg_ce)).item(),
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            
            output = self.model(input_ids)
            
            model_output = {
                'logits': output.logits,
                'phase_states': output.phase_states,
            }
            targets = {'token_ids': input_ids}
            context = {}
            
            batch_loss = 0.0
            batch_ce = 0.0
            for objective in self.objectives:
                result = objective(model_output, targets, context)
                batch_loss += result.loss.item() * objective.weight
                
                if objective.name == 'ce':
                    batch_ce = result.loss.item()
            
            total_loss += batch_loss
            total_ce_loss += batch_ce
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_ce = total_ce_loss / num_batches
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce,
            'perplexity': torch.exp(torch.tensor(avg_ce)).item(),
        }
    
    @torch.no_grad()
    def generate_sample(self, prompt: str = "The", max_tokens: int = 50) -> str:
        """Generate a sample from the model"""
        self.model.eval()
        
        # Tokenize prompt
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(prompt, return_tensors='pt')
        else:
            tokens = torch.tensor([self.tokenizer(prompt)['input_ids']])
        
        tokens = tokens.to(self.device)
        
        # Generate
        generated = self.model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
        
        # Decode
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text
    
    def save_checkpoint(self, name: str = 'checkpoint.pt'):
        """Save checkpoint"""
        path = self.checkpoint_dir / name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
        }, path)
        print(f"💾 Saved checkpoint to {path}")
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Training on {self.device}")
        print(f"   Model parameters: {self.model.count_parameters()['total']:,}")
        print(f"   Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"   Val batches: {len(self.val_loader)}")
        
        for epoch in range(self.config.training.max_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.training.max_epochs}")
            print('='*60)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"\n📊 Train | Loss: {train_metrics['loss']:.4f} | "
                  f"CE: {train_metrics['ce_loss']:.4f} | PPL: {train_metrics['perplexity']:.2f}")
            
            # Validate
            if self.val_loader:
                val_metrics = self.validate()
                print(f"📊 Val   | Loss: {val_metrics['loss']:.4f} | "
                      f"CE: {val_metrics['ce_loss']:.4f} | PPL: {val_metrics['perplexity']:.2f}")
                
                # Save best
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
            
            # Generate sample
            try:
                sample = self.generate_sample("The quick brown", max_tokens=30)
                print(f"\n📝 Sample: {sample}")
            except Exception as e:
                print(f"   (Sample generation failed: {e})")
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Final checkpoint
        self.save_checkpoint('final_model.pt')
        print("\n✅ Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train v4 with real data')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'tinystories'],
                        help='Dataset to use')
    parser.add_argument('--size', type=str, default='small',
                        choices=['tiny', 'small', 'medium', 'large'],
                        help='Model size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--max_train_samples', type=int, default=10000, 
                        help='Max training samples (for quick tests)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (uses default if not set)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v4_real')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    # Speed optimization arguments
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile for speedup')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--no_pin_memory', action='store_true', help='Disable pin_memory')
    parser.add_argument('--no_cache', action='store_true', help='Disable token caching')
    parser.add_argument('--cache_dir', type=str, default='.cache/v4_tokens', help='Token cache directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("v4 Quantum Phase-Field LLM - Real Data Training")
    print("="*60)
    
    # Load tokenizer
    tokenizer = get_tokenizer('gpt2')
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
    
    # Load dataset
    if args.dataset == 'wikitext2':
        train_texts = get_wikitext2('train', max_samples=args.max_train_samples)
        val_texts = get_wikitext2('validation', max_samples=1000)
    elif args.dataset == 'tinystories':
        train_texts = get_tinystories('train', max_samples=args.max_train_samples)
        val_texts = get_tinystories('validation', max_samples=1000)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create dataloaders with speed optimizations
    train_loader, val_loader = create_dataloaders(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )
    
    # Create config
    config = get_default_config(args.size)
    config.vocab_size = vocab_size
    config.training.max_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.checkpoint_dir = args.checkpoint_dir
    
    # Speed options
    config.training.compile_model = args.compile
    config.training.compile_mode = args.compile_mode
    config.training.num_workers = args.num_workers
    config.training.pin_memory = not args.no_pin_memory
    config.training.use_token_cache = not args.no_cache
    
    if args.lr:
        config.training.learning_rate = args.lr
    
    print(f"\n📋 Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Size: {args.size}")
    print(f"   Dim: {config.dim}")
    print(f"   Backbone layers: {config.backbone.num_layers}")
    print(f"   Banks: {list(config.banks.keys())}")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Max length: {args.max_length}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   LR: {config.training.learning_rate}")
    
    # Create model
    model = create_model(config=config)
    
    # Create trainer
    trainer = RealDataTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
    )
    
    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.global_step = checkpoint['global_step']
        print(f"📥 Resumed from {args.resume}")
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
