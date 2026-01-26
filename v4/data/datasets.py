#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Datasets: Real dataset integration

Supports:
- WikiText-2 (validation quality)
- TinyStories (good for small models)
- OpenWebText (large scale)
- Custom text files
"""

import os
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class V4Dataset(Dataset):
    """
    v4 Dataset for language modeling.
    
    Handles tokenization, chunking, and batching.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        stride: Optional[int] = None,
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer with encode/decode methods
            max_length: Maximum sequence length
            stride: Stride for overlapping chunks (None = no overlap)
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        
        # Tokenize and chunk all texts
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        """Tokenize texts and create fixed-length chunks"""
        print(f"📦 Preparing {len(self.texts)} texts...")
        
        for text in self.texts:
            if not text.strip():
                continue
            
            # Tokenize
            try:
                if hasattr(self.tokenizer, 'encode'):
                    tokens = self.tokenizer.encode(text)
                else:
                    tokens = self.tokenizer(text)['input_ids']
                
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.squeeze().tolist()
                
                # Handle single token case
                if isinstance(tokens, int):
                    tokens = [tokens]
                    
            except Exception as e:
                continue
            
            # Create chunks with stride
            if len(tokens) >= self.max_length:
                for i in range(0, len(tokens) - self.max_length + 1, self.stride):
                    chunk = tokens[i:i + self.max_length]
                    self.samples.append(chunk)
            elif len(tokens) > 10:  # Minimum viable length
                # Pad short sequences
                padded = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
                self.samples.append(padded)
        
        print(f"   Created {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.samples[idx]
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long)
        }


def get_wikitext2(split: str = 'train', max_samples: Optional[int] = None) -> List[str]:
    """
    Load WikiText-2 dataset.
    
    Args:
        split: 'train', 'validation', or 'test'
        max_samples: Limit number of samples (None = all)
    
    Returns:
        List of text strings
    """
    try:
        from datasets import load_dataset
        
        print(f"📥 Loading WikiText-2 ({split})...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        
        texts = [item['text'] for item in dataset if item['text'].strip()]
        
        if max_samples:
            texts = texts[:max_samples]
        
        print(f"   Loaded {len(texts)} texts")
        return texts
        
    except ImportError:
        print("⚠️ datasets library not installed")
        print("   Install with: pip install datasets")
        return _get_sample_texts(max_samples or 100)
    except Exception as e:
        print(f"⚠️ Failed to load WikiText-2: {e}")
        return _get_sample_texts(max_samples or 100)


def get_tinystories(split: str = 'train', max_samples: Optional[int] = None) -> List[str]:
    """
    Load TinyStories dataset (good for small models).
    
    Args:
        split: 'train' or 'validation'
        max_samples: Limit number of samples
    
    Returns:
        List of text strings
    """
    try:
        from datasets import load_dataset
        
        print(f"📥 Loading TinyStories ({split})...")
        dataset = load_dataset('roneneldan/TinyStories', split=split)
        
        texts = [item['text'] for item in dataset if item['text'].strip()]
        
        if max_samples:
            texts = texts[:max_samples]
        
        print(f"   Loaded {len(texts)} texts")
        return texts
        
    except ImportError:
        print("⚠️ datasets library not installed")
        return _get_sample_texts(max_samples or 100)
    except Exception as e:
        print(f"⚠️ Failed to load TinyStories: {e}")
        return _get_sample_texts(max_samples or 100)


def get_openwebtext(split: str = 'train', max_samples: Optional[int] = 10000) -> List[str]:
    """
    Load OpenWebText dataset (large scale).
    
    Args:
        split: 'train' only
        max_samples: Limit number of samples (recommended due to size)
    
    Returns:
        List of text strings
    """
    try:
        from datasets import load_dataset
        
        print(f"📥 Loading OpenWebText ({split}, max {max_samples})...")
        dataset = load_dataset('openwebtext', split=split, streaming=True)
        
        texts = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            if item['text'].strip():
                texts.append(item['text'])
        
        print(f"   Loaded {len(texts)} texts")
        return texts
        
    except ImportError:
        print("⚠️ datasets library not installed")
        return _get_sample_texts(max_samples or 100)
    except Exception as e:
        print(f"⚠️ Failed to load OpenWebText: {e}")
        return _get_sample_texts(max_samples or 100)


def _get_sample_texts(num_samples: int = 100) -> List[str]:
    """Generate sample texts for testing when real datasets unavailable"""
    print(f"📝 Generating {num_samples} sample texts...")
    
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models have revolutionized many fields.",
        "Transformers have become the dominant architecture for NLP.",
        "Phase-coded representations offer a new paradigm for language models.",
        "Quantum-inspired computing explores new computational approaches.",
        "Memory systems in neural networks enable long-term learning.",
        "Attention mechanisms allow models to focus on relevant information.",
        "The brain processes language through complex neural networks.",
    ]
    
    # Expand samples
    expanded = []
    for i in range(num_samples):
        base = samples[i % len(samples)]
        expanded.append(f"Sample {i}: {base} This is additional context for training.")
    
    return expanded


def create_dataloaders(
    train_texts: List[str],
    val_texts: Optional[List[str]] = None,
    tokenizer = None,
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_texts: Training texts
        val_texts: Validation texts (optional)
        tokenizer: Tokenizer (will use GPT-2 if None)
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: DataLoader workers
    
    Returns:
        (train_loader, val_loader) tuple
    """
    from .tokenizer import get_tokenizer
    
    if tokenizer is None:
        tokenizer = get_tokenizer('gpt2')
    
    # Create datasets
    train_dataset = V4Dataset(train_texts, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    
    val_loader = None
    if val_texts:
        val_dataset = V4Dataset(val_texts, tokenizer, max_length=max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    return train_loader, val_loader
