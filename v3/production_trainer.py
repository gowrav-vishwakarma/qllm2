#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production-Ready Brain-Inspired LLM Trainer
Features:
1. Real datasets (wikitext2, tinystories, openwebtext)
2. Proper tokenization (GPT-2 style)
3. Comprehensive progress tracking
4. Real-time monitoring
5. Checkpointing and resuming
6. Memory optimization
7. Experiment tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import psutil
import numpy as np
from collections import defaultdict, deque
import math

# Progress tracking
from tqdm import tqdm

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ Wandb not available, experiment tracking disabled")

# Tokenization
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available, using simple tokenizer")

# Import our brain-inspired components
from brain_inspired_llm import BrainInspiredLLM, create_brain_inspired_model
from brain_inspired_trainer import BrainInspiredTrainingSystem, ConsciousnessTrainer
from biologically_plausible_learning import BiologicallyPlausibleTrainer
from minimal_data_learning import MinimalDataLearningSystem

# Dataset integration
from dataset_integration import create_brain_inspired_data_loaders, get_dataset_configs

# BPE Tokenizer
from bpe_tokenizer import BPETokenizer, create_bpe_tokenizer_from_texts

class ProductionTokenizer:
    """Production-ready tokenizer with proper vocabulary management"""
    
    def __init__(self, vocab_size=50257, model_name="gpt2", texts=None):
        self.vocab_size = vocab_size
        self.model_name = model_name
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.vocab_size = len(self.tokenizer)
                print(f"✅ Loaded GPT-2 tokenizer with {self.vocab_size} tokens")
            except:
                print("⚠️ Failed to load GPT-2 tokenizer, using BPE tokenizer")
                self._create_bpe_tokenizer(texts)
        else:
            self._create_bpe_tokenizer(texts)
    
    def _create_bpe_tokenizer(self, texts=None):
        """Create a BPE tokenizer for better text processing"""
        print("🔄 Creating BPE tokenizer...")
        
        if texts is None:
            # Fallback to simple tokenizer if no texts provided
            self._create_simple_tokenizer()
            return
        
        # Create BPE tokenizer from texts
        self.bpe_tokenizer = create_bpe_tokenizer_from_texts(texts, vocab_size=self.vocab_size)
        self.vocab_size = len(self.bpe_tokenizer)
        print(f"✅ Created BPE tokenizer with {self.vocab_size} tokens")
    
    def _create_simple_tokenizer(self):
        """Create a simple character-level tokenizer as fallback"""
        print("🔄 Creating simple character-level tokenizer...")
        
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
        
        # Add extended ASCII
        for i in range(128, 256):
            char = chr(i)
            self.vocab[char] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"✅ Created simple tokenizer with {self.vocab_size} tokens")
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs"""
        if hasattr(self, 'tokenizer'):
            # Use GPT-2 tokenizer
            tokens = self.tokenizer.encode(text, max_length=max_length, truncation=True)
            return tokens
        elif hasattr(self, 'bpe_tokenizer'):
            # Use BPE tokenizer
            return self.bpe_tokenizer.encode(text, max_length=max_length)
        else:
            # Use simple tokenizer
            tokens = []
            for char in text[:max_length]:
                tokens.append(self.vocab.get(char, self.vocab['<unk>']))
            
            # Add EOS token
            if len(tokens) < max_length:
                tokens.append(self.vocab['<eos>'])
            
            return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        if hasattr(self, 'tokenizer'):
            # Use GPT-2 tokenizer
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        elif hasattr(self, 'bpe_tokenizer'):
            # Use BPE tokenizer
            return self.bpe_tokenizer.decode(token_ids)
        else:
            # Use simple tokenizer
            text = ""
            for token_id in token_ids:
                if token_id in self.idx_to_token:
                    char = self.idx_to_token[token_id]
                    if char not in ['<pad>', '<eos>', '<bos>']:
                        text += char
            return text
    
    def __len__(self):
        return self.vocab_size

class ProductionDataset(Dataset):
    """Production-ready dataset with proper tokenization and memory management"""
    
    def __init__(self, texts: List[str], tokenizer: ProductionTokenizer, 
                 max_length: int = 512, cache_size: int = 10000):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_size = cache_size
        
        # Cache for tokenized sequences
        self.token_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        print(f"📊 Dataset created with {len(texts)} texts")
        print(f"📊 Max length: {max_length}, Cache size: {cache_size}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Check cache first
        cache_key = f"{idx}_{self.max_length}"
        if cache_key in self.token_cache:
            self.cache_hits += 1
            return self.token_cache[cache_key]
        
        self.cache_misses += 1
        
        # Tokenize text
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        # Create input and target sequences
        if len(tokens) < 2:
            # Handle very short sequences
            tokens = tokens + [self.tokenizer.vocab.get('<pad>', 0)] * (2 - len(tokens))
        
        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Pad sequences
        if len(input_tokens) < self.max_length - 1:
            pad_length = self.max_length - 1 - len(input_tokens)
            input_tokens = torch.cat([input_tokens, torch.zeros(pad_length, dtype=torch.long)])
            target_tokens = torch.cat([target_tokens, torch.zeros(pad_length, dtype=torch.long)])
        
        result = {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'text_length': len(tokens),
            'text_idx': idx
        }
        
        # Cache result
        if len(self.token_cache) < self.cache_size:
            self.token_cache[cache_key] = result
        
        return result
    
    def get_cache_stats(self):
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.token_cache)
        }

class ProductionTrainer:
    """Production-ready trainer with comprehensive monitoring and optimization"""
    
    def __init__(self, model_config: Dict, training_config: Dict, 
                 output_dir: str = "checkpoints_production"):
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize tokenizer (will be updated with texts after dataset loading)
        self.tokenizer = None
        
        # Store model config for later use
        self.model_config = model_config
        self.training_config = training_config
        
        # Model and optimizer will be created after tokenizer is ready
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'memory_usage': [],
            'consciousness_metrics': [],
            'biological_metrics': []
        }
        
        # Performance monitoring
        self.step_times = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        
        # Setup wandb if available
        self._setup_wandb()
        
        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger.info(f"🚀 Production trainer initialized")
        self.logger.info(f"🖥️ Using device: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"🖥️ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def _initialize_model_and_tokenizer(self, texts: List[str]):
        """Initialize model and tokenizer after loading texts"""
        # Create tokenizer with texts
        self.tokenizer = ProductionTokenizer(
            vocab_size=self.model_config.get('vocab_size', 50257),
            texts=texts
        )
        
        # Update model config with actual vocab size
        self.model_config['vocab_size'] = len(self.tokenizer)
        
        # Create model
        self.model = create_brain_inspired_model(
            vocab_size=self.model_config['vocab_size'],
            dim=self.model_config['dim'],
            num_layers=self.model_config['num_layers']
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.training_config['num_epochs'],
            eta_min=self.training_config['learning_rate'] * 0.1
        )
        
        self.logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"📊 Vocab size: {len(self.tokenizer)}")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.output_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_wandb(self):
        """Setup Weights & Biases for experiment tracking"""
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="brain-inspired-llm",
                    config={
                        'model_config': self.model_config,
                        'training_config': self.training_config
                    },
                    name=f"brain_llm_{int(time.time())}"
                )
                self.use_wandb = True
                self.logger.info("✅ Wandb initialized")
            except:
                self.use_wandb = False
                self.logger.info("⚠️ Wandb initialization failed")
        else:
            self.use_wandb = False
            self.logger.info("⚠️ Wandb not available")
    
    def create_data_loaders(self, dataset_configs: List[Dict]) -> Tuple[DataLoader, DataLoader]:
        """Create production data loaders with real datasets"""
        self.logger.info("📚 Creating production data loaders...")
        
        # Load real datasets
        train_texts, val_texts = self._load_real_datasets(dataset_configs)
        
        # Initialize model and tokenizer with loaded texts
        if self.model is None:
            self._initialize_model_and_tokenizer(train_texts + val_texts)
        
        # Create datasets
        train_dataset = ProductionDataset(
            texts=train_texts,
            tokenizer=self.tokenizer,
            max_length=self.training_config['max_length'],
            cache_size=self.training_config.get('cache_size', 10000)
        )
        
        val_dataset = ProductionDataset(
            texts=val_texts,
            tokenizer=self.tokenizer,
            max_length=self.training_config['max_length'],
            cache_size=self.training_config.get('cache_size', 5000)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=self.training_config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 4),
            pin_memory=True,
            drop_last=False
        )
        
        self.logger.info(f"📊 Train batches: {len(train_loader)}")
        self.logger.info(f"📊 Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def _load_real_datasets(self, dataset_configs: List[Dict]) -> Tuple[List[str], List[str]]:
        """Load real datasets for training"""
        self.logger.info("📥 Loading real datasets...")
        
        train_texts = []
        val_texts = []
        
        for config in dataset_configs:
            dataset_name = config['name']
            self.logger.info(f"📥 Loading {dataset_name}...")
            
            try:
                if dataset_name == 'wikitext2':
                    texts = self._load_wikitext2(config)
                elif dataset_name == 'tinystories':
                    texts = self._load_tinystories(config)
                elif dataset_name == 'openwebtext':
                    texts = self._load_openwebtext(config)
                else:
                    self.logger.warning(f"⚠️ Unknown dataset: {dataset_name}")
                    continue
                
                # Split into train/val
                split_idx = int(len(texts) * 0.9)
                train_texts.extend(texts[:split_idx])
                val_texts.extend(texts[split_idx:])
                
                self.logger.info(f"✅ Loaded {len(texts)} texts from {dataset_name}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load {dataset_name}: {e}")
                continue
        
        # Fallback to sample data if no datasets loaded
        if not train_texts:
            self.logger.warning("⚠️ No datasets loaded, using sample data")
            train_texts, val_texts = self._create_sample_data()
        
        self.logger.info(f"📊 Total train texts: {len(train_texts)}")
        self.logger.info(f"📊 Total val texts: {len(val_texts)}")
        
        return train_texts, val_texts
    
    def _load_wikitext2(self, config: Dict) -> List[str]:
        """Load WikiText-2 dataset"""
        try:
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            texts = [item['text'] for item in dataset if len(item['text'].strip()) > 50]
            return texts[:config.get('max_samples', 10000)]
        except ImportError:
            self.logger.warning("⚠️ Datasets library not available")
            return []
    
    def _load_tinystories(self, config: Dict) -> List[str]:
        """Load TinyStories dataset"""
        try:
            from datasets import load_dataset
            dataset = load_dataset('roneneldan/TinyStories', split='train')
            texts = [item['text'] for item in dataset]
            return texts[:config.get('max_samples', 5000)]
        except ImportError:
            self.logger.warning("⚠️ Datasets library not available")
            return []
    
    def _load_openwebtext(self, config: Dict) -> List[str]:
        """Load OpenWebText dataset (subset)"""
        try:
            from datasets import load_dataset
            dataset = load_dataset('openwebtext', split='train')
            texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
            return texts[:config.get('max_samples', 2000)]
        except ImportError:
            self.logger.warning("⚠️ Datasets library not available")
            return []
    
    def _create_sample_data(self) -> Tuple[List[str], List[str]]:
        """Create sample data as fallback"""
        sample_texts = [
            "The brain-inspired language model uses consciousness mechanisms to process information and learn from minimal data.",
            "Memory consolidation and retrieval are key components of human-like learning in artificial intelligence systems.",
            "Spiking neurons provide event-driven processing similar to biological neural networks in the human brain.",
            "Hebbian learning rules enable neurons to strengthen connections through co-activation and temporal correlation.",
            "Short-term and long-term memory systems work together for effective information storage and retrieval.",
            "Developmental plasticity allows neural networks to adapt and grow over time based on experience.",
            "Minimal data learning enables systems to learn from very few examples, just like human learning.",
            "Consciousness awareness helps focus attention on important information and filter out noise.",
            "Biologically plausible learning avoids backpropagation for more realistic and efficient learning.",
            "Event-driven processing is more efficient than traditional continuous processing in neural networks."
        ]
        
        # Generate more samples
        train_texts = []
        val_texts = []
        
        for i in range(1000):
            # Combine 2-3 sample texts
            num_texts = np.random.randint(2, 4)
            selected = np.random.choice(sample_texts, num_texts, replace=True)
            combined = " ".join(selected)
            train_texts.append(combined)
        
        for i in range(200):
            num_texts = np.random.randint(1, 3)
            selected = np.random.choice(sample_texts, num_texts, replace=True)
            combined = " ".join(selected)
            val_texts.append(combined)
        
        return train_texts, val_texts
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop with comprehensive monitoring"""
        self.logger.info("🚀 Starting production training...")
        
        # Create progress bars
        epoch_pbar = tqdm(range(self.training_config['num_epochs']), 
                         desc="Epochs", position=0)
        
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch_pbar)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            self._save_checkpoint(train_metrics, val_metrics)
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train_loss': f"{train_metrics['loss']:.4f}",
                'val_loss': f"{val_metrics['loss']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        self.logger.info("✅ Training completed!")
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader, epoch_pbar) -> Dict:
        """Train for one epoch with progress tracking"""
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'perplexity': 0.0,
            'consciousness_awareness': 0.0,
            'memory_usage': 0.0,
            'learning_efficiency': 0.0,
            'step_time': 0.0
        }
        
        # Create batch progress bar
        batch_pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}", 
                         position=1, leave=False)
        
        for batch_idx, batch in enumerate(batch_pbar):
            step_start_time = time.time()
            
            # Get input and target
            input_tokens = batch['input_tokens'].to(self.device)
            target_tokens = batch['target_tokens'].to(self.device)
            
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
            
            # Update metrics
            step_time = time.time() - step_start_time
            self.step_times.append(step_time)
            
            # Get consciousness state
            consciousness_state = self.model.get_consciousness_state(input_tokens)
            memory_stats = self.model.get_memory_stats()
            
            # Update epoch metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['perplexity'] += torch.exp(loss).item()
            epoch_metrics['consciousness_awareness'] += consciousness_state['consciousness_weights'].mean().item()
            epoch_metrics['memory_usage'] += memory_stats['learning_efficiency']
            epoch_metrics['learning_efficiency'] += consciousness_state['consciousness_weights'].mean().item() * memory_stats['learning_efficiency']
            epoch_metrics['step_time'] += step_time
            
            self.current_step += 1
            
            # Update batch progress bar
            if batch_idx % 10 == 0:
                # Get memory usage
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
                gpu_percent = (gpu_mem / gpu_total * 100) if gpu_total > 0 else 0
                
                batch_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'ppl': f"{torch.exp(loss).item():.2f}",
                    'step_time': f"{step_time:.3f}s",
                    'gpu_mem': f"{gpu_mem:.1f}GB/{gpu_total:.1f}GB ({gpu_percent:.1f}%)"
                })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/perplexity': torch.exp(loss).item(),
                    'train/consciousness_awareness': consciousness_state['consciousness_weights'].mean().item(),
                    'train/memory_usage': memory_stats['learning_efficiency'],
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/step_time': step_time
                })
        
        # Average metrics
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        
        val_metrics = {
            'loss': 0.0,
            'perplexity': 0.0,
            'consciousness_awareness': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", position=2, leave=False):
                input_tokens = batch['input_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                
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
        
        # Average metrics
        num_batches = len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log training metrics"""
        # Update training history
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        self.training_history['consciousness_metrics'].append(train_metrics['consciousness_awareness'])
        
        # Log to console
        self.logger.info(f"Epoch {self.current_epoch+1}:")
        self.logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        self.logger.info(f"  Train PPL: {train_metrics['perplexity']:.2f}, Val PPL: {val_metrics['perplexity']:.2f}")
        self.logger.info(f"  Consciousness: {train_metrics['consciousness_awareness']:.4f}")
        self.logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        self.logger.info(f"  Step Time: {train_metrics['step_time']:.3f}s")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'epoch': self.current_epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/perplexity': train_metrics['perplexity'],
                'train/consciousness_awareness': train_metrics['consciousness_awareness'],
                'train/memory_usage': train_metrics['memory_usage'],
                'train/learning_efficiency': train_metrics['learning_efficiency'],
                'val/loss': val_metrics['loss'],
                'val/perplexity': val_metrics['perplexity'],
                'val/consciousness_awareness': val_metrics['consciousness_awareness'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
    
    def _save_checkpoint(self, train_metrics: Dict, val_metrics: Dict):
        """Save model checkpoint"""
        # Save best model
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            
            checkpoint = {
                'epoch': self.current_epoch,
                'step': self.current_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'model_config': self.model_config,
                'training_config': self.training_config,
                'training_history': self.training_history,
                'best_val_loss': self.best_val_loss,
                'tokenizer_vocab_size': len(self.tokenizer)
            }
            
            checkpoint_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"💾 Best model saved: {checkpoint_path}")
        
        # Save regular checkpoint
        if (self.current_epoch + 1) % 5 == 0:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch+1}.pt"
            checkpoint = {
                'epoch': self.current_epoch,
                'step': self.current_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'model_config': self.model_config,
                'training_config': self.training_config,
                'training_history': self.training_history,
                'best_val_loss': self.best_val_loss
            }
            torch.save(checkpoint, checkpoint_path)
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
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
    parser = argparse.ArgumentParser(description='Production Brain-Inspired LLM Training')
    
    # Model config
    parser.add_argument('--dim', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    
    # Training config
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Data config
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--cache_size', type=int, default=10000, help='Dataset cache size')
    
    # Output config
    parser.add_argument('--output_dir', type=str, default='checkpoints_production', help='Output directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    
    args = parser.parse_args()
    
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
        'grad_clip': args.grad_clip,
        'num_workers': args.num_workers,
        'cache_size': args.cache_size
    }
    
    # Create trainer
    trainer = ProductionTrainer(model_config, training_config, args.output_dir)
    
    # Dataset configurations
    dataset_configs = [
        {'name': 'wikitext2', 'max_samples': 10000},
        {'name': 'tinystories', 'max_samples': 5000},
        {'name': 'openwebtext', 'max_samples': 2000}
    ]
    
    # Create data loaders
    train_loader, val_loader = trainer.create_data_loaders(dataset_configs)
    
    # Train model
    training_history = trainer.train(train_loader, val_loader)
    
    # Test generation
    print("\n🎯 Testing text generation...")
    test_prompts = [
        "The brain-inspired language model",
        "Artificial intelligence is",
        "Memory consolidation helps",
        "Consciousness mechanisms enable",
        "Spiking neurons provide"
    ]
    
    for prompt in test_prompts:
        generated = trainer.generate_text(prompt, max_length=50, temperature=0.7)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print()
    
    print("🎉 Production training completed!")
    return trainer, training_history

if __name__ == "__main__":
    trainer, history = main()
