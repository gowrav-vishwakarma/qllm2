#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain-Inspired LLM Training Script
Trains the v3 brain-inspired language model on real datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
import psutil

# Import our brain-inspired components
from brain_inspired_llm import BrainInspiredLLM, create_brain_inspired_model
from brain_inspired_trainer import BrainInspiredTrainingSystem, ConsciousnessTrainer
from biologically_plausible_learning import BiologicallyPlausibleTrainer
from minimal_data_learning import MinimalDataLearningSystem
from dataset_integration import create_brain_inspired_data_loaders, get_dataset_configs

class SimpleTextDataset(Dataset):
    """Simple text dataset for training"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Simple character-level tokenization
        tokens = [ord(c) for c in text[:self.max_length]]
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        # Create input and target (shifted by 1)
        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_tokens, target_tokens

def create_sample_data(num_samples=1000, min_length=50, max_length=200):
    """Create sample training data"""
    print("📝 Creating sample training data...")
    
    # Sample texts for training
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a simple sentence for training our brain-inspired language model.",
        "Artificial intelligence is revolutionizing the way we think about computing and problem solving.",
        "The brain-inspired approach to machine learning mimics how humans learn and process information.",
        "Consciousness and awareness are key components of human-like artificial intelligence systems.",
        "Memory consolidation and retrieval mechanisms are essential for effective learning.",
        "Spiking neurons and event-driven processing provide efficient computation similar to biological systems.",
        "Hebbian learning rules state that neurons that fire together wire together.",
        "Short-term and long-term memory systems work together to store and retrieve information.",
        "Developmental plasticity allows neural networks to adapt and grow over time.",
        "Minimal data learning enables systems to learn from very few examples, just like humans."
    ]
    
    # Generate more samples by combining and varying the base texts
    texts = []
    for i in range(num_samples):
        # Randomly combine 2-4 base texts
        num_texts = np.random.randint(2, 5)
        selected_texts = np.random.choice(sample_texts, num_texts, replace=True)
        
        # Combine with random separators
        combined = " ".join(selected_texts)
        
        # Randomly truncate to desired length
        if len(combined) > max_length:
            start = np.random.randint(0, len(combined) - max_length)
            combined = combined[start:start + max_length]
        
        # Ensure minimum length
        if len(combined) < min_length:
            combined += " " + " ".join(np.random.choice(sample_texts, 1))
        
        texts.append(combined)
    
    print(f"✅ Created {len(texts)} training samples")
    return texts

def train_brain_inspired_model(
    model_config: Dict,
    training_config: Dict,
    data_config: Dict,
    output_dir: str = "checkpoints_brain_inspired"
):
    """Train the brain-inspired model"""
    
    print("🧠 BRAIN-INSPIRED LANGUAGE MODEL TRAINING")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    print(f"🏗️ Creating brain-inspired model...")
    print(f"   📊 Vocab size: {model_config['vocab_size']}")
    print(f"   📊 Dimension: {model_config['dim']}")
    print(f"   📊 Layers: {model_config['num_layers']}")
    
    model = create_brain_inspired_model(
        vocab_size=model_config['vocab_size'],
        dim=model_config['dim'],
        num_layers=model_config['num_layers']
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create training system
    training_system = BrainInspiredTrainingSystem(
        vocab_size=model_config['vocab_size'],
        dim=model_config['dim'],
        num_layers=model_config['num_layers']
    )
    
    # Create data loaders using brain-inspired dataset integration
    dataset_configs = get_dataset_configs()
    
    train_loader, val_loader = create_brain_inspired_data_loaders(
        dataset_configs=dataset_configs,
        batch_size=training_config['batch_size'],
        max_length=data_config['max_length'],
        consciousness_aware=True,
        use_v2_datasets=True
    )
    
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")
    print(f"📊 Batch size: {training_config['batch_size']}")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   📚 Epochs: {training_config['num_epochs']}")
    print(f"   🧠 Learning mode: {training_config['learning_mode']}")
    
    start_time = time.time()
    
    training_system.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_config['num_epochs'],
        learning_mode=training_config['learning_mode']
    )
    
    training_time = time.time() - start_time
    
    print(f"\n⏱️ Training completed in {training_time:.2f} seconds")
    
    # Test generation
    print(f"\n🎯 Testing text generation...")
    generated_texts = []
    for i in range(5):
        prompt = "The brain-inspired"
        generated = training_system.generate_text(prompt, max_length=50, temperature=0.7)
        generated_texts.append(generated)
        print(f"   Generated {i+1}: {generated[:100]}...")
    
    # Save model
    model_path = os.path.join(output_dir, "brain_inspired_model.pt")
    torch.save({
        'model_state_dict': training_system.model.state_dict(),
        'model_config': model_config,
        'training_config': training_config,
        'training_summary': training_system.get_training_summary()
    }, model_path)
    
    print(f"💾 Model saved to {model_path}")
    
    # Save training results
    results = {
        'model_config': model_config,
        'training_config': training_config,
        'data_config': data_config,
        'training_time': training_time,
        'training_summary': training_system.get_training_summary(),
        'generated_texts': generated_texts,
        'performance_metrics': {
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'memory_usage': psutil.virtual_memory().used / 1024**3,
            'training_efficiency': training_time / training_config['num_epochs']
        }
    }
    
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📊 Results saved to {results_path}")
    
    return training_system, results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Brain-Inspired Language Model')
    parser.add_argument('--vocab_size', type=int, default=256, help='Vocabulary size')
    parser.add_argument('--dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_mode', type=str, default='hybrid', 
                       choices=['consciousness', 'biological', 'minimal_data', 'hybrid'],
                       help='Learning mode')
    parser.add_argument('--num_train_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--num_val_samples', type=int, default=200, help='Number of validation samples')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='checkpoints_brain_inspired', help='Output directory')
    
    args = parser.parse_args()
    
    # Configuration
    model_config = {
        'vocab_size': args.vocab_size,
        'dim': args.dim,
        'num_layers': args.num_layers
    }
    
    training_config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_mode': args.learning_mode
    }
    
    data_config = {
        'num_train_samples': args.num_train_samples,
        'num_val_samples': args.num_val_samples,
        'min_length': 50,
        'max_length': args.max_length
    }
    
    # Train model
    training_system, results = train_brain_inspired_model(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        output_dir=args.output_dir
    )
    
    print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Model parameters: {results['performance_metrics']['model_parameters']:,}")
    print(f"⏱️ Training time: {results['training_time']:.2f}s")
    print(f"🧠 Learning mode: {training_config['learning_mode']}")
    print(f"💾 Memory usage: {results['performance_metrics']['memory_usage']:.2f} GB")
    
    return training_system, results

if __name__ == "__main__":
    training_system, results = main()
