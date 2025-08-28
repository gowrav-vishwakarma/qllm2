#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-efficient dataset handling for Quantum-Inspired LLM
Uses streaming and chunked processing to minimize RAM usage
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, List, Tuple, Iterator
from datasets import load_dataset

def collate_fn(batch):
    """Custom collate function for streaming dataset"""
    # Each item in batch is a tuple (chunk, target)
    chunks = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return torch.stack(chunks), torch.stack(targets)

class StreamingByteDataset(IterableDataset):
    """Streaming dataset that processes text in chunks without loading everything into RAM"""
    def __init__(self, dataset_name: str, split: str, seq_length: int, 
                 max_samples: Optional[int] = None, buffer_size: int = 10000):
        self.dataset_name = dataset_name
        self.split = split
        self.seq_length = seq_length
        self.max_samples = max_samples
        self.buffer_size = buffer_size
        self.samples_processed = 0
        
        # Initialize dataset iterator
        self._init_dataset_iter()
        
        # Initialize buffer
        self.buffer = []
        self.buffer_position = 0
        
    def _init_dataset_iter(self):
        """Initialize or reset the dataset iterator"""
        if self.dataset_name == "wikitext2":
            self.dataset_iter = load_dataset("wikitext", "wikitext-2-raw-v1", split=self.split, streaming=True).iter(batch_size=1)
        elif self.dataset_name == "tinystories":
            self.dataset_iter = load_dataset("roneneldan/TinyStories", split=self.split, streaming=True).iter(batch_size=1)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
    def _fill_buffer(self):
        """Fill the buffer with text from the streaming dataset"""
        self.buffer = []
        self.buffer_position = 0
        
        try:
            # Get a batch of examples
            for _ in range(self.buffer_size):
                example = next(self.dataset_iter)
                text = example["text"][0]  # Extract text from batch
                if text.strip():  # Skip empty texts
                    self.buffer.append(text)
        except StopIteration:
            # If we run out of data, reset the iterator and try again
            print("🔄 Resetting streaming dataset iterator for new epoch...")
            self._init_dataset_iter()
            try:
                for _ in range(self.buffer_size):
                    example = next(self.dataset_iter)
                    text = example["text"][0]
                    if text.strip():
                        self.buffer.append(text)
            except StopIteration:
                pass  # If still no data, we're done
        
        # Shuffle the buffer to introduce randomness
        if self.buffer:
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
            text = self.buffer[self.buffer_position]
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
        
        # Process texts into chunks with strict limits
        self.chunks = []
        self.targets = []
        total_chunks = 0
        
        # Use a hard limit on total chunks to prevent explosion
        max_total_chunks = max_chunks if max_chunks is not None else 5000
        
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
    
    def __len__(self):
        return len(self.chunks_tensor)
    
    def __getitem__(self, idx):
        return (
            self.chunks_tensor[idx],
            self.targets_tensor[idx]
        )

def build_loaders(dataset_name: str, seq_length: int, batch_size: int, 
                 max_samples: Optional[int] = None, streaming: bool = True,
                 num_workers: int = 4, val_max_chunks: int = 5000) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation data loaders with memory efficiency"""
    
    # For validation, use a much smaller fixed dataset
    if dataset_name == "wikitext2":
        val_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        val_texts = [ex["text"] for ex in val_texts if ex["text"].strip()]
        val_dataset = MemoryEfficientByteDataset(
            dataset_name, "validation", seq_length, 
            max_samples=min(100, len(val_texts)),  # Only use 100 texts for validation
            max_chunks=val_max_chunks  # Hard limit on validation chunks
        )
    elif dataset_name == "tinystories":
        val_texts = load_dataset("roneneldan/TinyStories", split="validation")
        val_texts = [ex["text"] for ex in val_texts]
        val_dataset = MemoryEfficientByteDataset(
            dataset_name, "validation", seq_length, 
            max_samples=min(100, len(val_texts)),  # Only use 100 texts for validation
            max_chunks=val_max_chunks  # Hard limit on validation chunks
        )
    
    # For training, use streaming if requested
    if streaming:
        train_dataset = StreamingByteDataset(
            dataset_name, "train", seq_length, max_samples, buffer_size=5000
        )
    else:
        train_dataset = MemoryEfficientByteDataset(
            dataset_name, "train", seq_length, max_samples
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