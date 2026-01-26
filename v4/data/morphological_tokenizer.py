#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MorphologicalTokenizer v1: Data-driven root+affix tokenizer

Key design principles:
1. NO hardcoded language-specific rules
2. Learns root and modifier vocabularies from corpus statistics
3. Works across scripts by operating on unicode text
4. Emits (root_id, prefix_id, suffix_id) per token

The tokenizer uses a 2-stage approach:
- Stage A: Learn "root-like" subword units (core meaning carriers)
- Stage B: Learn prefix and suffix modifier vocabularies

Encoding a word:
- Find best parse among: (prefix, root), (root, suffix), (prefix, root, suffix), (root only)
- Fallback: (root=whole_word, prefix=<null>, suffix=<null>)
"""

import json
import re
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import torch


# Special token IDs
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
NULL_AFFIX = "<null>"  # For when there's no prefix/suffix


@dataclass
class MorphVocab:
    """Vocabulary for roots or affixes"""
    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.token_to_id)
    
    def add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]
    
    def get_id(self, token: str, default: Optional[int] = None) -> int:
        return self.token_to_id.get(token, default if default is not None else self.token_to_id.get(UNK_TOKEN, 0))
    
    def get_token(self, idx: int) -> str:
        return self.id_to_token.get(idx, UNK_TOKEN)


@dataclass
class MorphologicalTokenizerConfig:
    """Configuration for MorphologicalTokenizer"""
    root_vocab_size: int = 16000
    prefix_vocab_size: int = 2000
    suffix_vocab_size: int = 2000
    min_root_len: int = 2
    max_affix_len: int = 5
    min_freq: int = 5
    # BPE-like merging parameters
    num_merges: int = 8000


class MorphologicalTokenizer:
    """
    Data-driven morphological tokenizer.
    
    Learns root and affix vocabularies from corpus statistics.
    Outputs (root_id, prefix_id, suffix_id) per token.
    """
    
    def __init__(self, config: Optional[MorphologicalTokenizerConfig] = None):
        self.config = config or MorphologicalTokenizerConfig()
        
        # Initialize vocabularies
        self.root_vocab = MorphVocab()
        self.prefix_vocab = MorphVocab()
        self.suffix_vocab = MorphVocab()
        
        # Add special tokens
        for vocab in [self.root_vocab, self.prefix_vocab, self.suffix_vocab]:
            vocab.add_token(PAD_TOKEN)
            vocab.add_token(UNK_TOKEN)
            vocab.add_token(BOS_TOKEN)
            vocab.add_token(EOS_TOKEN)
        
        # Add null affix for prefix/suffix vocabs
        self.prefix_vocab.add_token(NULL_AFFIX)
        self.suffix_vocab.add_token(NULL_AFFIX)
        
        # Learned patterns from training
        self._trained = False
        self._prefix_patterns: Set[str] = set()
        self._suffix_patterns: Set[str] = set()
        
        # Special token IDs for easy access
        self.pad_token_id = self.root_vocab.get_id(PAD_TOKEN)
        self.unk_token_id = self.root_vocab.get_id(UNK_TOKEN)
        self.bos_token_id = self.root_vocab.get_id(BOS_TOKEN)
        self.eos_token_id = self.root_vocab.get_id(EOS_TOKEN)
        self.null_affix_id = self.prefix_vocab.get_id(NULL_AFFIX)
        
        # Word boundary regex (works across scripts)
        self._word_re = re.compile(r'\S+|\s+')
    
    @property
    def vocab_size(self) -> int:
        """Return root vocab size for compatibility"""
        return len(self.root_vocab)
    
    @property
    def prefix_vocab_size(self) -> int:
        return len(self.prefix_vocab)
    
    @property
    def suffix_vocab_size(self) -> int:
        return len(self.suffix_vocab)
    
    def train(self, texts: List[str], verbose: bool = True) -> None:
        """
        Train tokenizer on corpus.
        
        Stage A: Learn root vocabulary using frequency-based subword extraction
        Stage B: Learn prefix/suffix vocabularies from word-start/end patterns
        
        Args:
            texts: List of training texts
            verbose: Print progress
        """
        if verbose:
            print("🧬 Training MorphologicalTokenizer...")
        
        # Step 1: Collect word frequencies
        word_freq = Counter()
        for text in texts:
            words = self._tokenize_to_words(text)
            word_freq.update(words)
        
        if verbose:
            print(f"   Collected {len(word_freq)} unique words")
        
        # Step 2: Extract character frequencies and learn subword patterns
        char_freq = Counter()
        for word, freq in word_freq.items():
            for char in word:
                char_freq[char] += freq
        
        # Step 3: Train root vocabulary (BPE-like merging)
        self._train_root_vocab(word_freq, verbose)
        
        # Step 4: Train affix vocabularies
        self._train_affix_vocabs(word_freq, verbose)
        
        self._trained = True
        
        if verbose:
            print(f"✅ Training complete:")
            print(f"   Root vocab: {len(self.root_vocab)}")
            print(f"   Prefix vocab: {len(self.prefix_vocab)}")
            print(f"   Suffix vocab: {len(self.suffix_vocab)}")
    
    def _tokenize_to_words(self, text: str) -> List[str]:
        """Split text into words (unicode-aware)"""
        tokens = self._word_re.findall(text)
        return [t for t in tokens if t.strip()]
    
    def _train_root_vocab(self, word_freq: Counter, verbose: bool = True) -> None:
        """Train root vocabulary using frequency-based subword extraction"""
        if verbose:
            print("   Training root vocabulary...")
        
        # Start with character-level vocab
        char_vocab = Counter()
        for word, freq in word_freq.items():
            for char in word:
                char_vocab[char] += freq
        
        # Add frequent characters to root vocab
        for char, freq in char_vocab.most_common():
            if len(self.root_vocab) >= self.config.root_vocab_size:
                break
            if freq >= self.config.min_freq:
                self.root_vocab.add_token(char)
        
        # Build up vocabulary with frequent substrings (simplified BPE)
        # Collect n-grams from words
        ngram_freq = Counter()
        for n in range(2, 8):  # 2 to 7 character n-grams
            for word, freq in word_freq.items():
                if len(word) >= n:
                    for i in range(len(word) - n + 1):
                        ngram = word[i:i+n]
                        ngram_freq[ngram] += freq
        
        # Add frequent n-grams as roots
        for ngram, freq in ngram_freq.most_common():
            if len(self.root_vocab) >= self.config.root_vocab_size:
                break
            if freq >= self.config.min_freq and len(ngram) >= self.config.min_root_len:
                self.root_vocab.add_token(ngram)
        
        # Add full words that are frequent enough
        for word, freq in word_freq.most_common():
            if len(self.root_vocab) >= self.config.root_vocab_size:
                break
            if freq >= self.config.min_freq:
                self.root_vocab.add_token(word)
    
    def _train_affix_vocabs(self, word_freq: Counter, verbose: bool = True) -> None:
        """Train prefix and suffix vocabularies from word patterns"""
        if verbose:
            print("   Training affix vocabularies...")
        
        prefix_freq = Counter()
        suffix_freq = Counter()
        
        # Collect potential affixes from words
        for word, freq in word_freq.items():
            if len(word) < self.config.min_root_len + 1:
                continue
            
            # Extract prefixes (1 to max_affix_len chars from start)
            for i in range(1, min(self.config.max_affix_len + 1, len(word) - self.config.min_root_len + 1)):
                prefix = word[:i]
                remaining = word[i:]
                # Only count if remaining is a known root or long enough
                if len(remaining) >= self.config.min_root_len:
                    prefix_freq[prefix] += freq
            
            # Extract suffixes (1 to max_affix_len chars from end)
            for i in range(1, min(self.config.max_affix_len + 1, len(word) - self.config.min_root_len + 1)):
                suffix = word[-i:]
                remaining = word[:-i]
                if len(remaining) >= self.config.min_root_len:
                    suffix_freq[suffix] += freq
        
        # Add frequent prefixes
        for prefix, freq in prefix_freq.most_common():
            if len(self.prefix_vocab) >= self.config.prefix_vocab_size:
                break
            if freq >= self.config.min_freq:
                self.prefix_vocab.add_token(prefix)
                self._prefix_patterns.add(prefix)
        
        # Add frequent suffixes
        for suffix, freq in suffix_freq.most_common():
            if len(self.suffix_vocab) >= self.config.suffix_vocab_size:
                break
            if freq >= self.config.min_freq:
                self.suffix_vocab.add_token(suffix)
                self._suffix_patterns.add(suffix)
    
    def _parse_word(self, word: str) -> Tuple[str, str, str]:
        """
        Parse a word into (prefix, root, suffix).
        
        Tries to find the best decomposition based on learned patterns.
        Returns (prefix, root, suffix) where prefix/suffix may be NULL_AFFIX.
        """
        if not word or len(word) < self.config.min_root_len:
            return (NULL_AFFIX, word if word else UNK_TOKEN, NULL_AFFIX)
        
        best_parse = (NULL_AFFIX, word, NULL_AFFIX)
        best_score = 0
        
        # Try to find prefix + root + suffix decomposition
        for prefix in self._prefix_patterns:
            if word.startswith(prefix) and len(word) > len(prefix) + self.config.min_root_len - 1:
                remaining = word[len(prefix):]
                
                # Check for suffix
                for suffix in self._suffix_patterns:
                    if remaining.endswith(suffix) and len(remaining) > len(suffix) + self.config.min_root_len - 1:
                        root = remaining[:-len(suffix)]
                        if len(root) >= self.config.min_root_len:
                            # Score: prefer longer affixes and known roots
                            score = len(prefix) + len(suffix)
                            if root in self.root_vocab.token_to_id:
                                score += 10
                            if score > best_score:
                                best_score = score
                                best_parse = (prefix, root, suffix)
                
                # Also try prefix + root (no suffix)
                if remaining in self.root_vocab.token_to_id or len(remaining) >= self.config.min_root_len:
                    score = len(prefix)
                    if remaining in self.root_vocab.token_to_id:
                        score += 10
                    if score > best_score:
                        best_score = score
                        best_parse = (prefix, remaining, NULL_AFFIX)
        
        # Try root + suffix (no prefix)
        for suffix in self._suffix_patterns:
            if word.endswith(suffix) and len(word) > len(suffix) + self.config.min_root_len - 1:
                root = word[:-len(suffix)]
                if len(root) >= self.config.min_root_len:
                    score = len(suffix)
                    if root in self.root_vocab.token_to_id:
                        score += 10
                    if score > best_score:
                        best_score = score
                        best_parse = (NULL_AFFIX, root, suffix)
        
        return best_parse
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
    ) -> Union[Tuple[List[int], List[int], List[int]], Dict[str, torch.Tensor]]:
        """
        Encode text to (root_ids, prefix_ids, suffix_ids).
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            truncation: Whether to truncate
            return_tensors: 'pt' for PyTorch tensors
            add_special_tokens: Whether to add BOS/EOS
        
        Returns:
            If return_tensors='pt': Dict with 'root_ids', 'prefix_ids', 'suffix_ids'
            Otherwise: Tuple of (root_ids, prefix_ids, suffix_ids)
        """
        words = self._tokenize_to_words(text)
        
        root_ids = []
        prefix_ids = []
        suffix_ids = []
        
        if add_special_tokens:
            root_ids.append(self.bos_token_id)
            prefix_ids.append(self.null_affix_id)
            suffix_ids.append(self.null_affix_id)
        
        for word in words:
            prefix, root, suffix = self._parse_word(word)
            
            root_id = self.root_vocab.get_id(root, self.unk_token_id)
            prefix_id = self.prefix_vocab.get_id(prefix, self.null_affix_id)
            suffix_id = self.suffix_vocab.get_id(suffix, self.null_affix_id)
            
            root_ids.append(root_id)
            prefix_ids.append(prefix_id)
            suffix_ids.append(suffix_id)
        
        if add_special_tokens:
            root_ids.append(self.eos_token_id)
            prefix_ids.append(self.null_affix_id)
            suffix_ids.append(self.null_affix_id)
        
        # Truncate if needed
        if max_length and truncation:
            root_ids = root_ids[:max_length]
            prefix_ids = prefix_ids[:max_length]
            suffix_ids = suffix_ids[:max_length]
        
        if return_tensors == 'pt':
            return {
                'root_ids': torch.tensor(root_ids).unsqueeze(0),
                'prefix_ids': torch.tensor(prefix_ids).unsqueeze(0),
                'suffix_ids': torch.tensor(suffix_ids).unsqueeze(0),
            }
        
        return root_ids, prefix_ids, suffix_ids
    
    def decode(
        self,
        root_ids: Union[List[int], torch.Tensor],
        prefix_ids: Optional[Union[List[int], torch.Tensor]] = None,
        suffix_ids: Optional[Union[List[int], torch.Tensor]] = None,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            root_ids: Root token IDs
            prefix_ids: Optional prefix token IDs
            suffix_ids: Optional suffix token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        if isinstance(root_ids, torch.Tensor):
            root_ids = root_ids.squeeze().tolist()
        if isinstance(prefix_ids, torch.Tensor):
            prefix_ids = prefix_ids.squeeze().tolist() if prefix_ids is not None else None
        if isinstance(suffix_ids, torch.Tensor):
            suffix_ids = suffix_ids.squeeze().tolist() if suffix_ids is not None else None
        
        # Handle single values
        if isinstance(root_ids, int):
            root_ids = [root_ids]
        if prefix_ids is not None and isinstance(prefix_ids, int):
            prefix_ids = [prefix_ids]
        if suffix_ids is not None and isinstance(suffix_ids, int):
            suffix_ids = [suffix_ids]
        
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        
        words = []
        for i, root_id in enumerate(root_ids):
            if skip_special_tokens and root_id in special_ids:
                continue
            
            root = self.root_vocab.get_token(root_id)
            
            # Reconstruct word
            word = ""
            if prefix_ids is not None and i < len(prefix_ids):
                prefix = self.prefix_vocab.get_token(prefix_ids[i])
                if prefix != NULL_AFFIX:
                    word += prefix
            
            word += root
            
            if suffix_ids is not None and i < len(suffix_ids):
                suffix = self.suffix_vocab.get_token(suffix_ids[i])
                if suffix != NULL_AFFIX:
                    word += suffix
            
            words.append(word)
        
        return " ".join(words)
    
    def __call__(
        self,
        text: str,
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Union[List[int], torch.Tensor]]:
        """Callable interface for compatibility"""
        result = self.encode(
            text,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        
        if isinstance(result, tuple):
            return {
                'root_ids': result[0],
                'prefix_ids': result[1],
                'suffix_ids': result[2],
            }
        return result
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to directory"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(path / "config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Save vocabularies
        with open(path / "root_vocab.json", 'w') as f:
            json.dump(self.root_vocab.token_to_id, f, indent=2, ensure_ascii=False)
        
        with open(path / "prefix_vocab.json", 'w') as f:
            json.dump(self.prefix_vocab.token_to_id, f, indent=2, ensure_ascii=False)
        
        with open(path / "suffix_vocab.json", 'w') as f:
            json.dump(self.suffix_vocab.token_to_id, f, indent=2, ensure_ascii=False)
        
        # Save patterns
        with open(path / "patterns.json", 'w') as f:
            json.dump({
                'prefix_patterns': list(self._prefix_patterns),
                'suffix_patterns': list(self._suffix_patterns),
            }, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MorphologicalTokenizer':
        """Load tokenizer from directory"""
        path = Path(path)
        
        # Load config
        with open(path / "config.json", 'r') as f:
            config_dict = json.load(f)
        config = MorphologicalTokenizerConfig(**config_dict)
        
        tokenizer = cls(config)
        
        # Load vocabularies
        with open(path / "root_vocab.json", 'r') as f:
            root_vocab = json.load(f)
        tokenizer.root_vocab.token_to_id = root_vocab
        tokenizer.root_vocab.id_to_token = {v: k for k, v in root_vocab.items()}
        
        with open(path / "prefix_vocab.json", 'r') as f:
            prefix_vocab = json.load(f)
        tokenizer.prefix_vocab.token_to_id = prefix_vocab
        tokenizer.prefix_vocab.id_to_token = {v: k for k, v in prefix_vocab.items()}
        
        with open(path / "suffix_vocab.json", 'r') as f:
            suffix_vocab = json.load(f)
        tokenizer.suffix_vocab.token_to_id = suffix_vocab
        tokenizer.suffix_vocab.id_to_token = {v: k for k, v in suffix_vocab.items()}
        
        # Load patterns
        with open(path / "patterns.json", 'r') as f:
            patterns = json.load(f)
        tokenizer._prefix_patterns = set(patterns['prefix_patterns'])
        tokenizer._suffix_patterns = set(patterns['suffix_patterns'])
        
        tokenizer._trained = True
        
        return tokenizer


def get_morphological_tokenizer(
    path: Optional[Union[str, Path]] = None,
    train_texts: Optional[List[str]] = None,
    config: Optional[MorphologicalTokenizerConfig] = None,
) -> MorphologicalTokenizer:
    """
    Get or create a morphological tokenizer.
    
    Args:
        path: Path to load from (if exists) or save to (after training)
        train_texts: Texts to train on (if not loading from path)
        config: Configuration for new tokenizer
    
    Returns:
        Trained MorphologicalTokenizer
    """
    if path and Path(path).exists():
        print(f"📂 Loading MorphologicalTokenizer from {path}")
        return MorphologicalTokenizer.load(path)
    
    tokenizer = MorphologicalTokenizer(config)
    
    if train_texts:
        tokenizer.train(train_texts)
        if path:
            print(f"💾 Saving MorphologicalTokenizer to {path}")
            tokenizer.save(path)
    else:
        print("⚠️ No training texts provided. Tokenizer is untrained.")
    
    return tokenizer
