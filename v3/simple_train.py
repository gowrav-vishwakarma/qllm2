#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Training Script for Brain-Inspired LLM
Minimal dependencies version for quick testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import argparse
from pathlib import Path
import json
import sys
import psutil
import gc

# Import our components
from brain_inspired_llm import create_brain_inspired_model

class SimpleTokenizer:
    """Simple character-level tokenizer"""
    
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        # Basic vocabulary: ASCII characters + special tokens
        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<eos>': 2,
            '<bos>': 3,
        }
        
        # Add ASCII characters
        for i in range(32, 127):  # Printable ASCII
            char = chr(i)
            self.vocab[char] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
    
    def __len__(self):
        return self.vocab_size
    
    def encode(self, text: str, max_length: int = 512):
        """Encode text to token IDs"""
        tokens = []
        for char in text[:max_length]:
            tokens.append(self.vocab.get(char, self.vocab['<unk>']))
        
        # Add EOS token
        if len(tokens) < max_length:
            tokens.append(self.vocab['<eos>'])
        
        return tokens
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        text = ""
        for token_id in token_ids:
            if token_id in self.idx_to_token:
                char = self.idx_to_token[token_id]
                if char not in ['<pad>', '<eos>', '<bos>']:
                    text += char
        return text

class SimpleDataset(Dataset):
    """Simple dataset for training"""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        # Create input and target sequences
        if len(tokens) < 2:
            tokens = tokens + [self.tokenizer.vocab['<pad>']] * (2 - len(tokens))
        
        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Pad sequences
        if len(input_tokens) < self.max_length - 1:
            pad_length = self.max_length - 1 - len(input_tokens)
            input_tokens = torch.cat([input_tokens, torch.zeros(pad_length, dtype=torch.long)])
            target_tokens = torch.cat([target_tokens, torch.zeros(pad_length, dtype=torch.long)])
        
        return input_tokens, target_tokens

class SimpleTrainer:
    """Simple trainer without external dependencies"""
    
    def __init__(self, model_config, training_config, output_dir="checkpoints_simple"):
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=model_config['vocab_size'])
        
        # Update model config with actual vocab size
        self.model_config['vocab_size'] = len(self.tokenizer)
        
        # Create model
        self.model = create_brain_inspired_model(
            vocab_size=self.model_config['vocab_size'],
            dim=self.model_config['dim'],
            num_layers=self.model_config['num_layers']
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01)
        )
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        print(f"🚀 Simple trainer initialized")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"📊 Vocab size: {len(self.tokenizer)}")
        
        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
            print(f"🖥️ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def get_memory_usage(self):
        """Get current memory usage"""
        memory_info = {}
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        memory_info['cpu_used_gb'] = cpu_memory.used / 1e9
        memory_info['cpu_total_gb'] = cpu_memory.total / 1e9
        memory_info['cpu_percent'] = cpu_memory.percent
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            memory_info['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            memory_info['gpu_percent'] = (memory_info['gpu_allocated_gb'] / memory_info['gpu_total_gb']) * 100
        else:
            memory_info['gpu_allocated_gb'] = 0
            memory_info['gpu_reserved_gb'] = 0
            memory_info['gpu_total_gb'] = 0
            memory_info['gpu_percent'] = 0
        
        return memory_info
    
    def create_sample_data(self, num_train=1000, num_val=200):
        """Create sample training data"""
        sample_texts = [
            "The brain-inspired language model uses consciousness mechanisms to process information.",
            "Memory consolidation and retrieval are key components of human-like learning.",
            "Spiking neurons provide event-driven processing similar to biological systems.",
            "Hebbian learning rules enable neurons to strengthen connections through co-activation.",
            "Short-term and long-term memory systems work together for effective information storage.",
            "Developmental plasticity allows neural networks to adapt and grow over time.",
            "Minimal data learning enables systems to learn from very few examples.",
            "Consciousness awareness helps focus attention on important information.",
            "Biologically plausible learning avoids backpropagation for more realistic learning.",
            "Event-driven processing is more efficient than traditional continuous processing."
        ]
        
        # Generate more samples
        train_texts = []
        val_texts = []
        
        for i in range(num_train):
            # Combine 2-3 sample texts
            num_texts = torch.randint(2, 4, (1,)).item()
            selected = torch.randperm(len(sample_texts))[:num_texts]
            combined = " ".join([sample_texts[idx] for idx in selected])
            train_texts.append(combined)
        
        for i in range(num_val):
            num_texts = torch.randint(1, 3, (1,)).item()
            selected = torch.randperm(len(sample_texts))[:num_texts]
            combined = " ".join([sample_texts[idx] for idx in selected])
            val_texts.append(combined)
        
        return train_texts, val_texts
    
    def create_data_loaders(self):
        """Create data loaders"""
        print("📚 Creating data loaders...")
        
        # Create sample data
        train_texts, val_texts = self.create_sample_data()
        
        # Create datasets
        train_dataset = SimpleDataset(
            texts=train_texts,
            tokenizer=self.tokenizer,
            max_length=self.training_config['max_length']
        )
        
        val_dataset = SimpleDataset(
            texts=val_texts,
            tokenizer=self.tokenizer,
            max_length=self.training_config['max_length']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        print(f"📊 Train batches: {len(train_loader)}")
        print(f"📊 Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader):
        """Main training loop with progress tracking"""
        print("🚀 Starting training...")
        print(f"📊 Total epochs: {self.training_config['num_epochs']}")
        print(f"📊 Train batches per epoch: {len(train_loader)}")
        print(f"📊 Val batches per epoch: {len(val_loader)}")
        print("💡 Press Ctrl+C to stop training and save current progress")
        print("=" * 60)
        
        total_start_time = time.time()
        
        try:
            for epoch in range(self.training_config['num_epochs']):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                print(f"\n📚 EPOCH {epoch + 1}/{self.training_config['num_epochs']}")
                print("-" * 40)
                
                # Training phase
                train_metrics = self._train_epoch(train_loader)
                
                # Validation phase
                val_metrics = self._validate_epoch(val_loader)
                
                # Log metrics
                self._log_metrics(train_metrics, val_metrics)
                
                # Save checkpoint
                self._save_checkpoint(train_metrics, val_metrics)
                
                # Epoch timing
                epoch_time = time.time() - epoch_start_time
                print(f"⏱️ Epoch {epoch + 1} completed in {epoch_time:.1f}s")
                
                # Estimate remaining time
                if epoch > 0:
                    avg_epoch_time = (time.time() - total_start_time) / (epoch + 1)
                    remaining_epochs = self.training_config['num_epochs'] - epoch - 1
                    estimated_remaining = avg_epoch_time * remaining_epochs
                    print(f"🕐 Estimated remaining time: {estimated_remaining:.1f}s")
            
            total_time = time.time() - total_start_time
            print("\n" + "=" * 60)
            print("✅ Training completed successfully!")
            print(f"⏱️ Total training time: {total_time:.1f}s")
            print(f"📊 Average time per epoch: {total_time/self.training_config['num_epochs']:.1f}s")
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Training interrupted by user (Ctrl+C)")
            print("💾 Saving current progress...")
            
            # Save interrupted checkpoint
            checkpoint = {
                'epoch': self.current_epoch,
                'step': self.current_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_config': self.model_config,
                'training_config': self.training_config,
                'training_history': self.training_history,
                'best_val_loss': self.best_val_loss,
                'interrupted': True
            }
            
            checkpoint_path = self.output_dir / "interrupted_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Interrupted checkpoint saved to: {checkpoint_path}")
            
            total_time = time.time() - total_start_time
            print(f"⏱️ Training ran for {total_time:.1f}s before interruption")
            print(f"📊 Completed {self.current_epoch + 1} epochs")
        
        return self.training_history
    
    def _train_epoch(self, train_loader):
        """Train for one epoch with progress tracking"""
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'perplexity': 0.0,
            'consciousness_awareness': 0.0
        }
        
        num_batches = len(train_loader)
        batch_times = []
        
        print(f"🔄 Training: {num_batches} batches")
        
        for batch_idx, (input_tokens, target_tokens) in enumerate(train_loader):
            batch_start_time = time.time()
            
            # Move data to device
            input_tokens = input_tokens.to(self.device)
            target_tokens = target_tokens.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_tokens, training_step=self.current_step)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_tokens.view(-1),
                ignore_index=0  # Ignore padding tokens
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.get('grad_clip', 1.0)
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Get consciousness state
            consciousness_state = self.model.get_consciousness_state(input_tokens)
            
            # Update metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['perplexity'] += torch.exp(loss).item()
            epoch_metrics['consciousness_awareness'] += consciousness_state['consciousness_weights'].mean().item()
            
            self.current_step += 1
            
            # Track batch time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Progress indicator
            progress = (batch_idx + 1) / num_batches
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Print progress every 10 batches or at the end
            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                avg_batch_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                remaining_batches = num_batches - batch_idx - 1
                eta = remaining_batches * avg_batch_time
                
                # Get memory usage
                memory_info = self.get_memory_usage()
                
                print(f"\r  [{bar}] {progress*100:.1f}% | "
                      f"Batch {batch_idx+1}/{num_batches} | "
                      f"Loss: {loss.item():.4f} | "
                      f"ETA: {eta:.1f}s | "
                      f"GPU: {memory_info['gpu_allocated_gb']:.1f}GB/{memory_info['gpu_total_gb']:.1f}GB "
                      f"({memory_info['gpu_percent']:.1f}%) | "
                      f"CPU: {memory_info['cpu_percent']:.1f}%", end='', flush=True)
        
        print()  # New line after progress bar
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch with progress tracking"""
        self.model.eval()
        
        val_metrics = {
            'loss': 0.0,
            'perplexity': 0.0,
            'consciousness_awareness': 0.0
        }
        
        num_batches = len(val_loader)
        print(f"🔍 Validation: {num_batches} batches")
        
        with torch.no_grad():
            for batch_idx, (input_tokens, target_tokens) in enumerate(val_loader):
                # Move data to device
                input_tokens = input_tokens.to(self.device)
                target_tokens = target_tokens.to(self.device)
                
                # Forward pass
                logits = self.model(input_tokens)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_tokens.view(-1),
                    ignore_index=0
                )
                
                # Get consciousness state
                consciousness_state = self.model.get_consciousness_state(input_tokens)
                
                # Update metrics
                val_metrics['loss'] += loss.item()
                val_metrics['perplexity'] += torch.exp(loss).item()
                val_metrics['consciousness_awareness'] += consciousness_state['consciousness_weights'].mean().item()
                
                # Progress indicator
                progress = (batch_idx + 1) / num_batches
                bar_length = 20
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                    # Get memory usage
                    memory_info = self.get_memory_usage()
                    
                    print(f"\r  [{bar}] {progress*100:.1f}% | "
                          f"Batch {batch_idx+1}/{num_batches} | "
                          f"Loss: {loss.item():.4f} | "
                          f"GPU: {memory_info['gpu_allocated_gb']:.1f}GB "
                          f"({memory_info['gpu_percent']:.1f}%)", end='', flush=True)
        
        print()  # New line after progress bar
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def _log_metrics(self, train_metrics, val_metrics):
        """Log training metrics"""
        # Update training history
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        # Log to console
        print(f"Epoch {self.current_epoch+1}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Train PPL: {train_metrics['perplexity']:.2f}, Val PPL: {val_metrics['perplexity']:.2f}")
        print(f"  Consciousness: {train_metrics['consciousness_awareness']:.4f}")
        print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
    
    def _save_checkpoint(self, train_metrics, val_metrics):
        """Save model checkpoint"""
        # Save best model
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            
            checkpoint = {
                'epoch': self.current_epoch,
                'step': self.current_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_config': self.model_config,
                'training_config': self.training_config,
                'training_history': self.training_history,
                'best_val_loss': self.best_val_loss,
                'tokenizer_vocab_size': len(self.tokenizer)
            }
            
            checkpoint_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Best model saved: {checkpoint_path}")
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """Generate text using the trained model"""
        self.model.eval()
        
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        generated = tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.model(generated)
                
                # Get next token probabilities
                next_token_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0].tolist())
        return generated_text[len(prompt):]

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Simple Brain-Inspired LLM Training')
    
    # Model configuration
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--vocab_size', type=int, default=256, help='Vocabulary size')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='checkpoints_simple', help='Output directory')
    
    args = parser.parse_args()
    
    # Print configuration
    print("🧠 SIMPLE BRAIN-INSPIRED LLM TRAINING")
    print("=" * 50)
    print(f"📊 Model: {args.dim}D, {args.num_layers} layers, vocab={args.vocab_size}")
    print(f"📚 Training: {args.num_epochs} epochs, batch={args.batch_size}, lr={args.learning_rate}")
    print(f"📝 Data: {args.max_length} max length")
    print(f"💾 Output: {args.output_dir}")
    print("=" * 50)
    
    # Configuration
    model_config = {
        'dim': args.dim,
        'num_layers': args.num_layers,
        'vocab_size': args.vocab_size
    }
    
    training_config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip
    }
    
    # Create trainer
    trainer = SimpleTrainer(model_config, training_config, args.output_dir)
    
    # Create data loaders
    train_loader, val_loader = trainer.create_data_loaders()
    
    # Train model
    start_time = time.time()
    training_history = trainer.train(train_loader, val_loader)
    training_time = time.time() - start_time
    
    print(f"\n⏱️ Training completed in {training_time:.2f} seconds")
    print(f"📊 Final validation loss: {trainer.best_val_loss:.4f}")
    
    # Test generation
    print("\n🎯 Testing text generation...")
    test_prompts = [
        "The brain-inspired",
        "Artificial intelligence",
        "Memory consolidation",
        "Consciousness mechanisms",
        "Spiking neurons"
    ]
    
    for prompt in test_prompts:
        generated = trainer.generate_text(prompt, max_length=50, temperature=0.7)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()
    
    # Save results
    results = {
        'training_time': training_time,
        'final_val_loss': trainer.best_val_loss,
        'training_history': training_history,
        'model_config': model_config,
        'training_config': training_config,
        'args': vars(args)
    }
    
    results_path = Path(args.output_dir) / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {results_path}")
    print("🎉 Simple training completed!")
    
    return trainer, training_history

if __name__ == "__main__":
    trainer, history = main()
