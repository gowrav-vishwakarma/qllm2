#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalable dataset handling for Quantum-Inspired LLM
Supports streaming, multi-dataset, and dynamic memory management
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, List, Tuple, Iterator, Union, Dict
from datasets import load_dataset, concatenate_datasets
import psutil
import gc

def collate_fn(batch):
    """Custom collate function for streaming dataset"""
    # Each item in batch is a tuple (chunk, target)
    chunks = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.stack(chunks), torch.stack(targets)

class ScalableStreamingDataset(IterableDataset):
    """
    Scalable streaming dataset that can handle multiple datasets
    and dynamically adjust to available memory
    """
    def __init__(self, 
                 dataset_configs: List[Dict],  # List of dataset configurations
                 seq_length: int,
                 max_samples: Optional[int] = None,
                 buffer_size: int = 10000,
                 memory_limit_gb: float = 8.0,
                 shuffle_buffer: bool = True):
        """
        Args:
            dataset_configs: List of dicts with keys:
                - name: dataset name (wikitext2, tinystories, etc.)
                - split: dataset split (train, validation)
                - weight: sampling weight for this dataset
                - max_samples: max samples from this dataset (None = all)
            seq_length: sequence length for chunks
            max_samples: total max samples across all datasets
            buffer_size: size of the streaming buffer
            memory_limit_gb: memory limit in GB for dataset loading
            shuffle_buffer: whether to shuffle the buffer
        """
        self.dataset_configs = dataset_configs
        self.seq_length = seq_length
        self.max_samples = max_samples
        self.buffer_size = buffer_size
        self.memory_limit_gb = memory_limit_gb
        self.shuffle_buffer = shuffle_buffer
        self.samples_processed = 0
        
        # Calculate total weight for sampling
        self.total_weight = sum(config.get('weight', 1.0) for config in dataset_configs)
        
        # Initialize dataset iterators
        self._init_dataset_iters()
        
        # Initialize buffer
        self.buffer = []
        self.buffer_position = 0
        
    def _init_dataset_iters(self):
        """Initialize dataset iterators for all configured datasets"""
        self.dataset_iters = {}
        
        for config in self.dataset_configs:
            dataset_name = config['name']
            split = config['split']
            
            if dataset_name == "wikitext2":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, streaming=True)
            elif dataset_name == "tinystories":
                dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
            elif dataset_name == "openwebtext":
                dataset = load_dataset("Skylion007/openwebtext", split=split, streaming=True)
            elif dataset_name == "pile":
                dataset = load_dataset("EleutherAI/pile", split=split, streaming=True)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            self.dataset_iters[dataset_name] = dataset.iter(batch_size=1)
    
    def _get_available_memory_gb(self):
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / (1024**3)
    
    def _should_use_streaming(self, dataset_size_estimate: int) -> bool:
        """Determine if we should use streaming based on memory constraints"""
        available_memory = self._get_available_memory_gb()
        
        # Estimate memory needed for full dataset (rough estimate)
        # Each chunk is seq_length * 2 (input + target) * 4 bytes (int32)
        estimated_memory_gb = (dataset_size_estimate * self.seq_length * 2 * 4) / (1024**3)
        
        # Use streaming if estimated memory > available memory * 0.5
        return estimated_memory_gb > (available_memory * 0.5)
    
    def _fill_buffer(self):
        """Fill the buffer with text from multiple datasets using weighted sampling"""
        self.buffer = []
        self.buffer_position = 0
        
        # Sample datasets based on weights
        dataset_names = [config['name'] for config in self.dataset_configs]
        dataset_weights = [config.get('weight', 1.0) / self.total_weight for config in self.dataset_configs]
        
        try:
            for _ in range(self.buffer_size):
                # Sample a dataset based on weights
                chosen_dataset = random.choices(dataset_names, weights=dataset_weights)[0]
                
                # Get example from chosen dataset
                example = next(self.dataset_iters[chosen_dataset])
                text = example["text"][0] if isinstance(example["text"], list) else example["text"]
                
                if text.strip():  # Skip empty texts
                    self.buffer.append((text, chosen_dataset))
                    
        except StopIteration:
            # If we run out of data, reset all iterators
            print("🔄 Resetting streaming dataset iterators for new epoch...")
            self._init_dataset_iters()
            try:
                for _ in range(self.buffer_size):
                    chosen_dataset = random.choices(dataset_names, weights=dataset_weights)[0]
                    example = next(self.dataset_iters[chosen_dataset])
                    text = example["text"][0] if isinstance(example["text"], list) else example["text"]
                    if text.strip():
                        self.buffer.append((text, chosen_dataset))
            except StopIteration:
                pass  # If still no data, we're done
        
        # Shuffle the buffer if requested
        if self.shuffle_buffer and self.buffer:
            random.shuffle(self.buffer)
    
    def _process_text(self, text: str) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Process a single text into chunks"""
        # Convert to bytes
        bytes_data = list(text.encode('utf-8', errors='ignore'))
        
        # Generate chunks
        for i in range(0, len(bytes_data) - self.seq_length - 1):
            chunk = bytes_data[i:i+self.seq_length]
            target = bytes_data[i+1:i+self.seq_length+1]
            
            yield (
                torch.tensor(chunk, dtype=torch.long),
                torch.tensor(target, dtype=torch.long)
            )
            
            self.samples_processed += 1
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over the dataset"""
        # Reset samples processed for new epoch
        self.samples_processed = 0
        
        while True:
            # If buffer is empty or we've processed all of it, refill
            if self.buffer_position >= len(self.buffer):
                self._fill_buffer()
                if not self.buffer:  # No more data even after reset
                    break
                self.buffer_position = 0
            
            # Process current text
            text, dataset_name = self.buffer[self.buffer_position]
            self.buffer_position += 1
            
            # Yield chunks from this text
            for chunk, target in self._process_text(text):
                # Check max_samples limit for this epoch
                if self.max_samples is not None and self.samples_processed >= self.max_samples:
                    return  # End this epoch
                yield chunk, target

class MemoryEfficientByteDataset(Dataset):
    """Memory-efficient dataset for smaller datasets that can fit in RAM"""
    def __init__(self, dataset_name: str, split: str, seq_length: int, 
                 max_samples: Optional[int] = None, max_chunks: Optional[int] = None):
        self.seq_length = seq_length
        
        # Load dataset with limits
        if dataset_name == "wikitext2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [ex["text"] for ex in dataset if ex["text"].strip()]
        elif dataset_name == "tinystories":
            dataset = load_dataset("roneneldan/TinyStories", split=split)
            texts = [ex["text"] for ex in dataset]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Limit texts if requested
        if max_samples is not None:
            texts = texts[:max_samples]
        
        # Process texts into chunks with dynamic limits based on memory
        self.chunks = []
        self.targets = []
        total_chunks = 0
        
        # Dynamic chunk limit based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if max_chunks is not None:
            max_total_chunks = max_chunks
        else:
            # Estimate chunks based on available memory
            # Each chunk uses seq_length * 2 * 4 bytes
            estimated_chunks_per_gb = (1024**3) / (seq_length * 2 * 4)
            max_total_chunks = int(available_memory_gb * estimated_chunks_per_gb * 0.3)  # Use 30% of available memory
            max_total_chunks = max(1000, min(max_total_chunks, 100000))  # Between 1k and 100k chunks
        
        print(f"📊 Using dynamic chunk limit: {max_total_chunks} chunks based on {available_memory_gb:.1f}GB available memory")
        
        for text in texts:
            # Convert to bytes
            bytes_data = list(text.encode('utf-8', errors='ignore'))
            
            # Generate chunks with limit
            for i in range(0, len(bytes_data) - seq_length - 1):
                if total_chunks >= max_total_chunks:  # Stop if we've reached the limit
                    break
                    
                chunk = bytes_data[i:i+seq_length]
                target = bytes_data[i+1:i+seq_length+1]
                
                self.chunks.append(chunk)
                self.targets.append(target)
                total_chunks += 1
            
            if total_chunks >= max_total_chunks:  # Stop processing more texts
                break
        
        # Convert to tensors once
        self.chunks_tensor = torch.tensor(self.chunks, dtype=torch.long)
        self.targets_tensor = torch.tensor(self.targets, dtype=torch.long)
        
        print(f"📊 Created dataset with {len(self.chunks_tensor)} chunks from {len(texts)} texts")
        
        # Free memory
        del self.chunks
        del self.targets
        gc.collect()
    
    def __len__(self):
        return len(self.chunks_tensor)
    
    def __getitem__(self, idx):
        return (
            self.chunks_tensor[idx],
            self.targets_tensor[idx]
        )

def build_loaders(dataset_configs: Union[str, List[Dict]], seq_length: int, batch_size: int, 
                 max_samples: Optional[int] = None, streaming: bool = True,
                 num_workers: int = 4, val_max_chunks: int = 5000) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation data loaders with scalable architecture
    
    Args:
        dataset_configs: Either a string (single dataset) or list of dicts (multi-dataset)
        seq_length: sequence length
        batch_size: batch size
        max_samples: max samples across all datasets
        streaming: whether to use streaming (recommended for large datasets)
        num_workers: number of workers for data loading
        val_max_chunks: max chunks for validation
    """
    
    # Convert single dataset string to config format
    if isinstance(dataset_configs, str):
        dataset_configs = [{'name': dataset_configs, 'split': 'train', 'weight': 1.0}]
    
    # For validation, use a smaller subset of the same datasets
    val_configs = []
    for config in dataset_configs:
        val_config = config.copy()
        val_config['split'] = 'validation'
        # Handle None max_samples properly
        max_samples = config.get('max_samples', 1000)
        if max_samples is not None:
            val_config['max_samples'] = min(100, max_samples)  # Smaller validation set
        else:
            val_config['max_samples'] = 100  # Default validation size
        val_configs.append(val_config)
    
    # Build validation dataset
    if streaming:
        val_dataset = ScalableStreamingDataset(
            val_configs, seq_length, max_samples=val_max_chunks, 
            buffer_size=1000, memory_limit_gb=2.0
        )
    else:
        # For validation, use memory-efficient approach with first dataset only
        val_config = val_configs[0]
        val_dataset = MemoryEfficientByteDataset(
            val_config['name'], val_config['split'], seq_length, 
            max_samples=val_config.get('max_samples', 100), 
            max_chunks=val_max_chunks
        )
    
    # Build training dataset
    if streaming:
        train_dataset = ScalableStreamingDataset(
            dataset_configs, seq_length, max_samples, 
            buffer_size=10000, memory_limit_gb=8.0
        )
    else:
        # For non-streaming, use memory-efficient approach with first dataset only
        train_config = dataset_configs[0]
        train_dataset = MemoryEfficientByteDataset(
            train_config['name'], train_config['split'], seq_length, 
            max_samples=train_config.get('max_samples', max_samples)
        )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not streaming,  # No need to shuffle streaming dataset (we shuffle buffer)
        num_workers=num_workers if not streaming else 0,  # No multiprocessing for streaming
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, num_workers),  # Fewer workers for validation
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

# Example usage functions for different scenarios
def get_single_dataset_config(dataset_name: str, split: str = 'train', weight: float = 1.0, max_samples: Optional[int] = None) -> List[Dict]:
    """Get configuration for a single dataset"""
    return [{'name': dataset_name, 'split': split, 'weight': weight, 'max_samples': max_samples}]

def get_multi_dataset_config() -> List[Dict]:
    """Get configuration for multiple datasets with different weights"""
    return [
        {'name': 'wikitext2', 'split': 'train', 'weight': 0.4, 'max_samples': None},
        {'name': 'tinystories', 'split': 'train', 'weight': 0.3, 'max_samples': None},
        {'name': 'openwebtext', 'split': 'train', 'weight': 0.3, 'max_samples': 100000}
    ]

def get_large_scale_config() -> List[Dict]:
    """Get configuration for large-scale training with multiple datasets"""
    return [
        {'name': 'pile', 'split': 'train', 'weight': 0.5, 'max_samples': None},
        {'name': 'openwebtext', 'split': 'train', 'weight': 0.3, 'max_samples': None},
        {'name': 'wikitext2', 'split': 'train', 'weight': 0.2, 'max_samples': None}
    ]