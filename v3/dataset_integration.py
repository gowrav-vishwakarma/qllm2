#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Integration for Brain-Inspired LLM (v3)
Integrates with existing dataset system from v2
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import json

# Add v2 to path to import dataset system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v2'))

try:
    from datasets_qllm import ScalableStreamingDataset, create_dataset_configs
    DATASET_SYSTEM_AVAILABLE = True
except ImportError:
    DATASET_SYSTEM_AVAILABLE = False
    print("⚠️ v2 dataset system not available, using simple dataset")

class BrainInspiredDataset(Dataset):
    """Brain-inspired dataset that works with consciousness and memory systems"""
    
    def __init__(self, texts: List[str], tokenizer=None, max_length: int = 128, 
                 consciousness_aware: bool = True):
        """
        Args:
            texts: List of text samples
            tokenizer: Tokenizer function (if None, uses character-level)
            max_length: Maximum sequence length
            consciousness_aware: Whether to include consciousness metadata
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.consciousness_aware = consciousness_aware
        
        # Consciousness metadata
        self.consciousness_scores = []
        self.attention_weights = []
        self.memory_importance = []
        
        if consciousness_aware:
            self._compute_consciousness_metadata()
    
    def _compute_consciousness_metadata(self):
        """Compute consciousness-related metadata for each text"""
        print("🧠 Computing consciousness metadata...")
        
        for text in self.texts:
            # Simple consciousness scoring based on text properties
            # In a real implementation, this would use the consciousness layer
            
            # Length-based consciousness (longer texts might be more "conscious")
            length_score = min(len(text) / 100.0, 1.0)
            
            # Complexity-based consciousness (more unique words = more complex)
            unique_words = len(set(text.split()))
            total_words = len(text.split())
            complexity_score = unique_words / max(total_words, 1)
            
            # Coherence-based consciousness (repetition patterns)
            words = text.split()
            if len(words) > 1:
                repetition_score = 1.0 - (len(set(words)) / len(words))
            else:
                repetition_score = 0.0
            
            # Combined consciousness score
            consciousness_score = (length_score + complexity_score + (1 - repetition_score)) / 3.0
            self.consciousness_scores.append(consciousness_score)
            
            # Attention weights (simplified - focus on important words)
            attention_weights = self._compute_attention_weights(text)
            self.attention_weights.append(attention_weights)
            
            # Memory importance (how important this text is for memory)
            memory_importance = consciousness_score * 0.8 + 0.2  # Base importance
            self.memory_importance.append(memory_importance)
    
    def _compute_attention_weights(self, text: str) -> List[float]:
        """Compute attention weights for each character in text"""
        # Simple attention: higher weight for vowels, punctuation, and capital letters
        weights = []
        for char in text:
            if char.isupper():
                weight = 0.8
            elif char in 'aeiouAEIOU':
                weight = 0.6
            elif char in '.,!?;:':
                weight = 0.9
            elif char.isalpha():
                weight = 0.4
            else:
                weight = 0.2
            weights.append(weight)
        
        # Normalize weights
        if weights:
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]
        
        return weights
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text (character-level if no tokenizer provided)"""
        if self.tokenizer:
            return self.tokenizer(text)
        else:
            # Character-level tokenization
            return [ord(c) for c in text]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self._tokenize(text)
        
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        # Create input and target (shifted by 1)
        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long)
        
        result = {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
        }
        
        if self.consciousness_aware:
            # Add consciousness metadata
            consciousness_score = self.consciousness_scores[idx]
            attention_weights = self.attention_weights[idx]
            memory_importance = self.memory_importance[idx]
            
            # Pad attention weights to match sequence length
            if len(attention_weights) < self.max_length - 1:
                attention_weights.extend([0.0] * (self.max_length - 1 - len(attention_weights)))
            else:
                attention_weights = attention_weights[:self.max_length - 1]
            
            result.update({
                'consciousness_score': torch.tensor(consciousness_score, dtype=torch.float),
                'attention_weights': torch.tensor(attention_weights, dtype=torch.float),
                'memory_importance': torch.tensor(memory_importance, dtype=torch.float),
            })
        
        return result

def create_brain_inspired_data_loaders(
    dataset_configs: List[Dict],
    batch_size: int = 8,
    max_length: int = 128,
    consciousness_aware: bool = True,
    use_v2_datasets: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for brain-inspired training
    
    Args:
        dataset_configs: Dataset configurations
        batch_size: Batch size
        max_length: Maximum sequence length
        consciousness_aware: Whether to include consciousness metadata
        use_v2_datasets: Whether to use v2 dataset system if available
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    if use_v2_datasets and DATASET_SYSTEM_AVAILABLE:
        print("🔄 Using v2 dataset system...")
        return _create_v2_data_loaders(dataset_configs, batch_size, max_length, consciousness_aware)
    else:
        print("🔄 Using simple dataset system...")
        return _create_simple_data_loaders(batch_size, max_length, consciousness_aware)

def _create_v2_data_loaders(
    dataset_configs: List[Dict],
    batch_size: int,
    max_length: int,
    consciousness_aware: bool
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders using v2 dataset system"""
    
    # Create v2 streaming datasets
    train_dataset = ScalableStreamingDataset(
        dataset_configs=dataset_configs,
        seq_length=max_length,
        max_samples=10000,  # Limit for testing
        buffer_size=1000,
        memory_limit_gb=4.0
    )
    
    # Create validation dataset with different configs
    val_configs = []
    for config in dataset_configs:
        val_config = config.copy()
        val_config['split'] = 'validation' if 'validation' in val_config else 'test'
        val_configs.append(val_config)
    
    val_dataset = ScalableStreamingDataset(
        dataset_configs=val_configs,
        seq_length=max_length,
        max_samples=2000,  # Smaller validation set
        buffer_size=500,
        memory_limit_gb=2.0
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn_brain_inspired
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn_brain_inspired
    )
    
    return train_loader, val_loader

def _create_simple_data_loaders(
    batch_size: int,
    max_length: int,
    consciousness_aware: bool
) -> Tuple[DataLoader, DataLoader]:
    """Create simple data loaders with sample data"""
    
    # Create sample texts
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
    
    for i in range(1000):  # Training samples
        # Combine 2-3 sample texts
        num_texts = torch.randint(2, 4, (1,)).item()
        selected = torch.randperm(len(sample_texts))[:num_texts]
        combined = " ".join([sample_texts[idx] for idx in selected])
        train_texts.append(combined)
    
    for i in range(200):  # Validation samples
        num_texts = torch.randint(1, 3, (1,)).item()
        selected = torch.randperm(len(sample_texts))[:num_texts]
        combined = " ".join([sample_texts[idx] for idx in selected])
        val_texts.append(combined)
    
    # Create datasets
    train_dataset = BrainInspiredDataset(
        texts=train_texts,
        max_length=max_length,
        consciousness_aware=consciousness_aware
    )
    
    val_dataset = BrainInspiredDataset(
        texts=val_texts,
        max_length=max_length,
        consciousness_aware=consciousness_aware
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_brain_inspired
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_brain_inspired
    )
    
    return train_loader, val_loader

def collate_fn_brain_inspired(batch):
    """Custom collate function for brain-inspired datasets"""
    # Handle both v2 format and brain-inspired format
    if isinstance(batch[0], dict):
        # Brain-inspired format with consciousness metadata
        input_tokens = torch.stack([item['input_tokens'] for item in batch])
        target_tokens = torch.stack([item['target_tokens'] for item in batch])
        
        result = {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
        }
        
        # Add consciousness metadata if available
        if 'consciousness_score' in batch[0]:
            consciousness_scores = torch.stack([item['consciousness_score'] for item in batch])
            attention_weights = torch.stack([item['attention_weights'] for item in batch])
            memory_importance = torch.stack([item['memory_importance'] for item in batch])
            
            result.update({
                'consciousness_scores': consciousness_scores,
                'attention_weights': attention_weights,
                'memory_importance': memory_importance,
            })
        
        return result
    else:
        # v2 format (tensor, tensor)
        input_tokens = torch.stack([item[0] for item in batch])
        target_tokens = torch.stack([item[1] for item in batch])
        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
        }

def get_dataset_configs() -> List[Dict]:
    """Get default dataset configurations"""
    if DATASET_SYSTEM_AVAILABLE:
        try:
            return create_dataset_configs()
        except:
            pass
    
    # Fallback configurations
    return [
        {
            'name': 'wikitext2',
            'split': 'train',
            'weight': 1.0,
            'max_samples': 5000
        },
        {
            'name': 'tinystories',
            'split': 'train', 
            'weight': 1.0,
            'max_samples': 3000
        }
    ]

def test_dataset_integration():
    """Test the dataset integration"""
    print("🧪 Testing Brain-Inspired Dataset Integration...")
    
    # Test with simple datasets
    train_loader, val_loader = create_brain_inspired_data_loaders(
        dataset_configs=[],
        batch_size=4,
        max_length=64,
        consciousness_aware=True,
        use_v2_datasets=False
    )
    
    print(f"📊 Train batches: {len(train_loader)}")
    print(f"📊 Val batches: {len(val_loader)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"📊 Batch keys: {batch.keys()}")
    print(f"📊 Input shape: {batch['input_tokens'].shape}")
    print(f"📊 Target shape: {batch['target_tokens'].shape}")
    
    if 'consciousness_scores' in batch:
        print(f"📊 Consciousness scores shape: {batch['consciousness_scores'].shape}")
        print(f"📊 Attention weights shape: {batch['attention_weights'].shape}")
        print(f"📊 Memory importance shape: {batch['memory_importance'].shape}")
    
    print("✅ Dataset integration test completed!")
    return train_loader, val_loader

if __name__ == "__main__":
    test_dataset_integration()
