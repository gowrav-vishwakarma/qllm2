#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Training Script for Brain-Inspired LLM
Integrates with v2 dataset system and provides comprehensive training
"""

import os
import sys
import torch
import argparse
import time
from pathlib import Path

# Add v2 to path for dataset integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v2'))

# Import production trainer
from production_trainer import ProductionTrainer

def main():
    """Main training function with comprehensive configuration"""
    
    parser = argparse.ArgumentParser(description='Production Brain-Inspired LLM Training')
    
    # Model configuration
    parser.add_argument('--dim', type=int, default=768, 
                       help='Model dimension (default: 768)')
    parser.add_argument('--num_layers', type=int, default=12, 
                       help='Number of brain-inspired layers (default: 12)')
    parser.add_argument('--vocab_size', type=int, default=50257, 
                       help='Vocabulary size (default: 50257 for GPT-2)')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size (default: 8)')
    parser.add_argument('--num_epochs', type=int, default=10, 
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--max_length', type=int, default=512, 
                       help='Maximum sequence length (default: 512)')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--grad_clip', type=float, default=1.0, 
                       help='Gradient clipping (default: 1.0)')
    
    # Data configuration
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loader workers (default: 4)')
    parser.add_argument('--cache_size', type=int, default=10000, 
                       help='Dataset cache size (default: 10000)')
    parser.add_argument('--use_v2_datasets', action='store_true', 
                       help='Use v2 dataset system if available')
    
    # Dataset selection
    parser.add_argument('--datasets', nargs='+', 
                       default=['wikitext2', 'tinystories'],
                       choices=['wikitext2', 'tinystories', 'openwebtext'],
                       help='Datasets to use for training')
    parser.add_argument('--max_samples_per_dataset', type=int, default=10000,
                       help='Maximum samples per dataset (default: 10000)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='checkpoints_production', 
                       help='Output directory (default: checkpoints_production)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for tracking')
    parser.add_argument('--use_wandb', action='store_true', 
                       help='Use Weights & Biases for experiment tracking')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use (default: auto)')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training')
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Print configuration
    print("🧠 BRAIN-INSPIRED LLM PRODUCTION TRAINING")
    print("=" * 60)
    print(f"📊 Model: {args.dim}D, {args.num_layers} layers, vocab={args.vocab_size}")
    print(f"📚 Training: {args.num_epochs} epochs, batch={args.batch_size}, lr={args.learning_rate}")
    print(f"📝 Data: {args.max_length} max length, {args.datasets} datasets")
    print(f"💾 Output: {args.output_dir}")
    print("=" * 60)
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"🖥️ Using device: {device}")
    
    # Model configuration
    model_config = {
        'dim': args.dim,
        'num_layers': args.num_layers,
        'vocab_size': args.vocab_size,
        'device': device
    }
    
    # Training configuration
    training_config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'num_workers': args.num_workers,
        'cache_size': args.cache_size,
        'mixed_precision': args.mixed_precision,
        'use_v2_datasets': args.use_v2_datasets
    }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if args.experiment_name:
        output_dir = output_dir / args.experiment_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    print("🏗️ Creating production trainer...")
    trainer = ProductionTrainer(
        model_config=model_config,
        training_config=training_config,
        output_dir=str(output_dir)
    )
    
    # Setup experiment tracking
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project="brain-inspired-llm-production",
                name=args.experiment_name or f"brain_llm_{int(time.time())}",
                config={
                    'model_config': model_config,
                    'training_config': training_config,
                    'args': vars(args)
                }
            )
            print("✅ Wandb initialized")
        except ImportError:
            print("⚠️ Wandb not available, skipping experiment tracking")
    
    # Dataset configurations
    dataset_configs = []
    for dataset_name in args.datasets:
        config = {
            'name': dataset_name,
            'max_samples': args.max_samples_per_dataset,
            'split': 'train'
        }
        dataset_configs.append(config)
    
    print(f"📚 Using datasets: {[c['name'] for c in dataset_configs]}")
    
    # Create data loaders
    print("📥 Creating data loaders...")
    train_loader, val_loader = trainer.create_data_loaders(dataset_configs)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"🔄 Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.current_step = checkpoint['step']
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.training_history = checkpoint['training_history']
        print(f"✅ Resumed from epoch {trainer.current_epoch}")
    
    # Start training
    print("🚀 Starting training...")
    start_time = time.time()
    
    try:
        training_history = trainer.train(train_loader, val_loader)
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        print(f"📊 Final validation loss: {trainer.best_val_loss:.4f}")
        
        # Test generation
        print("\n🎯 Testing text generation...")
        test_prompts = [
            "The brain-inspired language model",
            "Artificial intelligence is revolutionizing",
            "Memory consolidation helps",
            "Consciousness mechanisms enable",
            "Spiking neurons provide"
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\nTest {i+1}:")
            print(f"Prompt: '{prompt}'")
            generated = trainer.generate_text(prompt, max_length=100, temperature=0.7)
            print(f"Generated: '{generated}'")
        
        # Save final results
        results = {
            'training_time': training_time,
            'final_val_loss': trainer.best_val_loss,
            'training_history': training_history,
            'model_config': model_config,
            'training_config': training_config,
            'args': vars(args)
        }
        
        results_path = output_dir / "final_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_path}")
        print(f"💾 Best model saved to: {output_dir / 'best_model.pt'}")
        
        return trainer, training_history
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("💾 Saving current checkpoint...")
        
        # Save interrupted checkpoint
        checkpoint = {
            'epoch': trainer.current_epoch,
            'step': trainer.current_step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'model_config': model_config,
            'training_config': training_config,
            'training_history': trainer.training_history,
            'best_val_loss': trainer.best_val_loss,
            'interrupted': True
        }
        
        checkpoint_path = output_dir / "interrupted_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Interrupted checkpoint saved to: {checkpoint_path}")
        
        return trainer, trainer.training_history
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    trainer, history = main()
