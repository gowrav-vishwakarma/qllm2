#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Tokenizer: GPT-2 tokenizer wrapper

Uses HuggingFace transformers for GPT-2 BPE tokenizer.
Falls back to HuggingFace `tokenizers` (fast BPE) if transformers not available.
Falls back to simple character-level if neither is available.
"""

from typing import List, Optional, Union
import torch


class SimpleTokenizer:
    """Fallback character-level tokenizer"""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
    
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


def get_tokenizer(name: str = 'gpt2'):
    """
    Get tokenizer by name.
    
    Args:
        name: 'gpt2' for GPT-2 BPE, 'simple' for character-level
    
    Returns:
        Tokenizer object with encode/decode methods
    """
    if name == 'simple':
        print("📝 Using simple character-level tokenizer")
        return SimpleTokenizer()
    
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
        return tokenizer
        
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ Failed to load {name} tokenizer: {e}")
        print("   Trying `tokenizers` backend...")

    # 2) Try tokenizers (already in your env via uv sync)
    try:
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download

        print(f"📝 Loading {name} tokenizer via `tokenizers`...")
        # GPT-2 tokenizer JSON is available on the Hub
        tok_json = hf_hub_download(repo_id=name, filename="tokenizer.json")
        tokenizer = Tokenizer.from_file(tok_json)

        class HFTokenizersWrapper:
            def __init__(self, tok: Tokenizer):
                self._tok = tok
                # GPT-2 doesn't have pad by default; align to EOS
                self.pad_token_id = 50256
                self.eos_token_id = 50256
                self.bos_token_id = 50256
                self.vocab_size = tok.get_vocab_size()

            def encode(self, text: str, max_length: Optional[int] = None,
                       truncation: bool = True, return_tensors: Optional[str] = None):
                ids = self._tok.encode(text).ids
                if max_length and truncation:
                    ids = ids[:max_length]
                if return_tensors == 'pt':
                    return torch.tensor(ids).unsqueeze(0)
                return ids

            def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                if token_ids and isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                return self._tok.decode(token_ids)

            def __call__(self, text: str, **kwargs) -> dict:
                return {"input_ids": self.encode(text, **kwargs)}

        wrapped = HFTokenizersWrapper(tokenizer)
        print(f"   Vocab size: {wrapped.vocab_size}")
        return wrapped

    except Exception as e:
        print("⚠️ transformers not installed, and `tokenizers` backend failed.")
        print(f"   Reason: {e}")
        print("   Falling back to simple character-level tokenizer (vocab_size=256).")
        return SimpleTokenizer()
