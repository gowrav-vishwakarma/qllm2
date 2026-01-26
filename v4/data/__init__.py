#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Data: Dataset and tokenizer integration

Supports two tokenizer modes:
1. GPT-2 BPE (baseline) - standard subword tokenization
2. Morphological (v4 innovation) - root + prefix + suffix decomposition
"""

from .datasets import (
    V4Dataset,
    MorphologicalDataset,
    create_dataloaders,
    get_wikitext2,
    get_tinystories,
)
from .tokenizer import get_tokenizer
from .morphological_tokenizer import (
    MorphologicalTokenizer,
    MorphologicalTokenizerConfig,
    get_morphological_tokenizer,
)

__all__ = [
    # Datasets
    'V4Dataset',
    'MorphologicalDataset',
    'create_dataloaders',
    'get_wikitext2',
    'get_tinystories',
    # Tokenizers
    'get_tokenizer',
    'MorphologicalTokenizer',
    'MorphologicalTokenizerConfig',
    'get_morphological_tokenizer',
]
