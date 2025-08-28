#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset handling for Quantum-Inspired LLM
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple
from datasets import load_dataset

class ByteDataset(Dataset):
    """Dataset that converts text to byte-level tokens"""
    def __init__(self, texts: List[str], seq_length: int):
        self.seq_length = seq_length
        
        # Concatenate all texts
        joined_text = "\n\n".join(texts)
        
        # Convert to bytes
        self.bytes_data = list(joined_text.encode('utf-8', errors='ignore'))
        
        # Create chunks
        self.chunks = []
        for i in range(0, len(self.bytes_data) - seq_length - 1):
            chunk = self.bytes_data[i:i+seq_length]
            target = self.bytes_data[i+1:i+seq_length+1]
            self.chunks.append((chunk, target))
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk, target = self.chunks[idx]
        return (
            torch.tensor(chunk, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )

def load_wikitext(split: str, max_samples: Optional[int] = None) -> List[str]:
    """Load wikitext dataset"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    texts = [ex["text"] for ex in dataset if ex["text"].strip()]
    
    if max_samples is not None:
        texts = texts[:max_samples]
    
    return texts

def load_tinystories(split: str, max_samples: Optional[int] = None) -> List[str]:
    """Load TinyStories dataset"""
    dataset = load_dataset("roneneldan/TinyStories", split=split)
    texts = [ex["text"] for ex in dataset]
    
    if max_samples is not None:
        texts = texts[:max_samples]
    
    return texts

def build_loaders(dataset_name: str, seq_length: int, batch_size: int, 
                 max_samples: Optional[int] = None, streaming: bool = False,
                 num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation data loaders"""
    
    # Load dataset
    if dataset_name == "wikitext2":
        train_texts = load_wikitext("train", max_samples)
        val_texts = load_wikitext("validation", None)
    elif dataset_name == "tinystories":
        train_texts = load_tinystories("train", max_samples)
        val_texts = load_tinystories("validation", None)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create datasets
    train_dataset = ByteDataset(train_texts, seq_length)
    val_dataset = ByteDataset(val_texts, seq_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader