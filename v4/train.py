#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Training Script: Simple training loop for Quantum Phase-Field LLM

Usage:
    python train.py --size small --epochs 10
    python train.py --config my_config.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from v4.model import QuantumPhaseFieldLLM, create_model, ModelOutput
from v4.core.config import V4Config, get_default_config, load_config
from v4.core.registry import get_registry
from v4.core.interfaces import ObjectiveResult


class SimpleTextDataset(Dataset):
    """Simple dataset for testing - generates random token sequences"""
    
    def __init__(self, num_samples: int = 1000, seq_len: int = 128, vocab_size: int = 50257):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate random data for testing
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'input_ids': self.data[idx]}


class Trainer:
    """Simple trainer for v4 model"""
    
    def __init__(
        self,
        model: QuantumPhaseFieldLLM,
        config: V4Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.max_epochs,
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Create objectives from config
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
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_metrics: Dict[str, float] = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            
            # Forward pass
            with autocast(enabled=self.scaler is not None):
                output = self.model(input_ids)
                
                # Compute all objectives
                total_batch_loss = torch.tensor(0.0, device=self.device)
                batch_metrics = {}
                
                model_output = {
                    'logits': output.logits,
                    'phase_states': output.phase_states,
                }
                targets = {'token_ids': input_ids}
                context = {'coupling_loss': output.coupling_loss}
                
                for objective in self.objectives:
                    result = objective(model_output, targets, context)
                    weighted_loss = result.loss * objective.weight
                    total_batch_loss = total_batch_loss + weighted_loss
                    
                    for key, value in result.metrics.items():
                        batch_metrics[f"{objective.name}/{key}"] = value
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += total_batch_loss.item()
            for key, value in batch_metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % self.config.training.log_every == 0:
                avg_loss = total_loss / num_batches
                print(f"  Step {self.global_step} | Loss: {avg_loss:.4f}")
        
        # Average metrics
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        avg_metrics['loss'] = total_loss / num_batches
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_metrics: Dict[str, float] = {}
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            
            output = self.model(input_ids)
            
            model_output = {
                'logits': output.logits,
                'phase_states': output.phase_states,
            }
            targets = {'token_ids': input_ids}
            context = {'coupling_loss': output.coupling_loss}
            
            batch_loss = 0.0
            for objective in self.objectives:
                result = objective(model_output, targets, context)
                batch_loss += result.loss.item() * objective.weight
                
                for key, value in result.metrics.items():
                    total_metrics[f"{objective.name}/{key}"] = total_metrics.get(
                        f"{objective.name}/{key}", 0.0
                    ) + value
            
            total_loss += batch_loss
            num_batches += 1
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        avg_metrics['loss'] = total_loss / num_batches
        
        return avg_metrics
    
    def save_checkpoint(self, name: str = 'checkpoint.pt'):
        """Save a checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
        }
        
        path = self.checkpoint_dir / name
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load a checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from {path}")
    
    def train(self):
        """Main training loop"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters()}")
        
        for epoch in range(self.config.training.max_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.config.training.max_epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch()
            train_time = time.time() - epoch_start
            
            print(f"Train | Loss: {train_metrics['loss']:.4f} | Time: {train_time:.1f}s")
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                print(f"Val   | Loss: {val_metrics['loss']:.4f}")
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
            
            # Step scheduler
            self.scheduler.step()
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Final checkpoint
        self.save_checkpoint('final_model.pt')
        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description='Train v4 Quantum Phase-Field LLM')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--size', type=str, default='tiny', 
                        choices=['tiny', 'small', 'medium', 'large'],
                        help='Model size preset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v4',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config(args.size)
    
    # Apply CLI overrides
    config.training.max_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.checkpoint_dir = args.checkpoint_dir
    
    print("=" * 60)
    print("v4 Quantum Phase-Field LLM Training")
    print("=" * 60)
    print(f"Size: {args.size}")
    print(f"Dim: {config.dim}")
    print(f"Backbone layers: {config.backbone.num_layers}")
    print(f"Banks: {list(config.banks.keys())}")
    print(f"Epochs: {config.training.max_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print("=" * 60)
    
    # Create model
    model = create_model(config=config)
    
    # Create datasets (using simple random data for testing)
    # TODO: Replace with real dataset integration
    train_dataset = SimpleTextDataset(
        num_samples=1000,
        seq_len=128,
        vocab_size=config.vocab_size
    )
    val_dataset = SimpleTextDataset(
        num_samples=100,
        seq_len=128,
        vocab_size=config.vocab_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create trainer
    trainer = Trainer(model, config, train_loader, val_loader)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
