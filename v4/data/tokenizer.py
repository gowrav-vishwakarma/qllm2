#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Tokenizer: Unified tokenizer interface

Supports:
1. GPT-2 BPE (baseline) - uses HuggingFace transformers or tokenizers
2. Morphological (v4) - data-driven root+affix decomposition
3. Simple character-level - fallback

Use get_tokenizer(name, tokenizer_type='bpe') for standard BPE
Use get_tokenizer(name, tokenizer_type='morphological', ...) for morphological
"""

from typing import List, Optional, Union, Any, Dict
from pathlib import Path
import torch


class SimpleTokenizer:
    """Fallback character-level tokenizer"""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.tokenizer_type = 'simple'
    
    def encode(self, text: str, max_length: Optional[int] = None, 
               truncation: bool = True, return_tensors: Optional[str] = None) -> Union[List[int], torch.Tensor]:
        # Character-level encoding (ASCII)
        tokens = [min(ord(c), self.vocab_size - 1) for c in text]
        
        if max_length and truncation:
            tokens = tokens[:max_length]
        
        if return_tensors == 'pt':
            return torch.tensor(tokens).unsqueeze(0)
        return tokens
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Handle nested lists
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        
        chars = []
        for t in token_ids:
            if skip_special_tokens and t in [self.pad_token_id, self.eos_token_id, self.bos_token_id]:
                continue
            if 32 <= t < 127:  # Printable ASCII
                chars.append(chr(t))
            else:
                chars.append('?')
        return ''.join(chars)
    
    def __call__(self, text: str, **kwargs) -> dict:
        tokens = self.encode(text, **kwargs)
        return {'input_ids': tokens}


class BPETokenizerWrapper:
    """Wrapper for BPE tokenizers with unified interface"""
    
    def __init__(self, tokenizer: Any, vocab_size: int):
        self._tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.tokenizer_type = 'bpe'
        
        # Copy token IDs
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', 50256)
        self.eos_token_id = getattr(tokenizer, 'eos_token_id', 50256)
        self.bos_token_id = getattr(tokenizer, 'bos_token_id', 50256)
    
    def encode(self, text: str, max_length: Optional[int] = None,
               truncation: bool = True, return_tensors: Optional[str] = None,
               **kwargs) -> Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode text to token IDs"""
        if hasattr(self._tokenizer, 'encode'):
            ids = self._tokenizer.encode(text)
            if hasattr(ids, 'ids'):  # tokenizers library
                ids = ids.ids
        else:
            ids = self._tokenizer(text)['input_ids']
        
        if max_length and truncation:
            ids = ids[:max_length]
        
        if return_tensors == 'pt':
            return torch.tensor(ids).unsqueeze(0)
        return ids
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], 
               skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self._tokenizer.decode(token_ids)
    
    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        return {'input_ids': self.encode(text, **kwargs)}


def get_tokenizer(
    name: str = 'gpt2',
    tokenizer_type: str = 'bpe',
    morph_path: Optional[Union[str, Path]] = None,
    morph_train_texts: Optional[List[str]] = None,
    morph_config: Optional[Any] = None,
) -> Any:
    """
    Get tokenizer by name and type.
    
    Args:
        name: Model name for BPE ('gpt2') or 'simple' for char-level
        tokenizer_type: 'bpe' for standard BPE, 'morphological' for root+affix
        morph_path: Path to load/save morphological tokenizer
        morph_train_texts: Texts to train morphological tokenizer on
        morph_config: Config for morphological tokenizer
    
    Returns:
        Tokenizer object with encode/decode methods
    """
    if tokenizer_type == 'morphological':
        from .morphological_tokenizer import get_morphological_tokenizer
        return get_morphological_tokenizer(
            path=morph_path,
            train_texts=morph_train_texts,
            config=morph_config,
        )
    
    if name == 'simple':
        print("📝 Using simple character-level tokenizer")
        return SimpleTokenizer()
    
    # Try to load BPE tokenizer
    return _get_bpe_tokenizer(name)


def _get_bpe_tokenizer(name: str = 'gpt2') -> BPETokenizerWrapper:
    """Load BPE tokenizer from various backends"""
    
    # 1) Prefer transformers (full-featured)
    try:
        from transformers import GPT2Tokenizer
        
        print(f"📝 Loading {name} tokenizer from HuggingFace...")
        tokenizer = GPT2Tokenizer.from_pretrained(name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print(f"   Vocab size: {tokenizer.vocab_size}")
        return BPETokenizerWrapper(tokenizer, tokenizer.vocab_size)
        
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ Failed to load {name} tokenizer: {e}")
        print("   Trying `tokenizers` backend...")

    # 2) Try tokenizers library
    try:
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download

        print(f"📝 Loading {name} tokenizer via `tokenizers`...")
        tok_json = hf_hub_download(repo_id=name, filename="tokenizer.json")
        tokenizer = Tokenizer.from_file(tok_json)

        class TokenizersWrapper:
            def __init__(self, tok: Tokenizer):
                self._tok = tok
                self.pad_token_id = 50256
                self.eos_token_id = 50256
                self.bos_token_id = 50256

            def encode(self, text: str):
                return self._tok.encode(text)

            def decode(self, token_ids) -> str:
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                if token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                return self._tok.decode(token_ids)

        wrapped = TokenizersWrapper(tokenizer)
        vocab_size = tokenizer.get_vocab_size()
        print(f"   Vocab size: {vocab_size}")
        return BPETokenizerWrapper(wrapped, vocab_size)

    except Exception as e:
        print("⚠️ transformers not installed, and `tokenizers` backend failed.")
        print(f"   Reason: {e}")
        print("   Falling back to simple character-level tokenizer (vocab_size=256).")
        return SimpleTokenizer()
