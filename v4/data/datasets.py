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
    Supports token caching for faster training.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        stride: Optional[int] = None,
        cache_path: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer with encode/decode methods
            max_length: Maximum sequence length
            stride: Stride for overlapping chunks (None = no overlap)
            cache_path: Path to cache tokenized samples
            use_cache: Whether to use caching
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        self.cache_path = cache_path
        self.use_cache = use_cache
        
        # Tokenize and chunk all texts
        self.samples = []
        self._tensor_cache: Optional[torch.Tensor] = None  # Pre-computed tensor cache
        self._prepare_samples()
    
    def _prepare_samples(self):
        """Tokenize texts and create fixed-length chunks"""
        # Try to load from cache
        if self.use_cache and self.cache_path and Path(self.cache_path).exists():
            print(f"📂 Loading cached tokens from {self.cache_path}")
            try:
                cache = torch.load(self.cache_path)
                self.samples = cache['samples']
                if 'tensor_cache' in cache:
                    self._tensor_cache = cache['tensor_cache']
                print(f"   Loaded {len(self.samples)} cached samples")
                return
            except Exception as e:
                print(f"   Cache load failed: {e}, reprocessing...")
        
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
        
        # Create tensor cache for faster __getitem__
        if self.samples:
            print("   Creating tensor cache...")
            self._tensor_cache = torch.tensor(self.samples, dtype=torch.long)
        
        # Save to cache
        if self.use_cache and self.cache_path and self.samples:
            print(f"💾 Saving token cache to {self.cache_path}")
            try:
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'samples': self.samples,
                    'tensor_cache': self._tensor_cache,
                }, self.cache_path)
            except Exception as e:
                print(f"   Cache save failed: {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Use tensor cache for faster access
        if self._tensor_cache is not None:
            return {'input_ids': self._tensor_cache[idx]}
        
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
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders with speed optimizations.
    
    Args:
        train_texts: Training texts
        val_texts: Validation texts (optional)
        tokenizer: Tokenizer (will use GPT-2 if None)
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: DataLoader workers (default: 4 for speed)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        prefetch_factor: Number of batches to prefetch per worker (default: 2)
        use_cache: Whether to cache tokenized samples (default: True)
        cache_dir: Directory for token cache files
    
    Returns:
        (train_loader, val_loader) tuple
    """
    import torch.cuda
    from .tokenizer import get_tokenizer
    
    if tokenizer is None:
        tokenizer = get_tokenizer('gpt2')
    
    # Determine cache paths
    train_cache = None
    val_cache = None
    if use_cache and cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        train_cache = f"{cache_dir}/train_tokens.pt"
        val_cache = f"{cache_dir}/val_tokens.pt"
    
    # Auto-detect optimal settings
    use_cuda = torch.cuda.is_available()
    actual_pin_memory = pin_memory and use_cuda
    actual_num_workers = num_workers if use_cuda else 0  # Workers only help with GPU
    
    # Create datasets with caching
    train_dataset = V4Dataset(
        train_texts, 
        tokenizer, 
        max_length=max_length,
        cache_path=train_cache,
        use_cache=use_cache,
    )
    
    # Configure DataLoader for speed
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': actual_num_workers,
        'pin_memory': actual_pin_memory,
        'drop_last': True,
        'persistent_workers': actual_num_workers > 0,  # Keep workers alive between epochs
    }
    
    # Add prefetch_factor only when using workers
    if actual_num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    
    val_loader = None
    if val_texts:
        val_dataset = V4Dataset(
            val_texts, 
            tokenizer, 
            max_length=max_length,
            cache_path=val_cache,
            use_cache=use_cache,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **{**loader_kwargs, 'drop_last': False},
        )
    
    # Log dataloader configuration
    print(f"⚡ DataLoader config: workers={actual_num_workers}, "
          f"pin_memory={actual_pin_memory}, prefetch={prefetch_factor}")
    
    return train_loader, val_loader
