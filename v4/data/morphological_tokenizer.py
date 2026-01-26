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
    prefix_vocab_size: int = 512  # Smaller for quality
    suffix_vocab_size: int = 512  # Smaller for quality
    min_root_len: int = 3  # Increased from 2 to avoid single-char roots
    max_affix_len: int = 5
    min_freq: int = 5
    # BPE-like merging parameters
    num_merges: int = 8000
    # Parse cache size (0 to disable)
    parse_cache_size: int = 100000
    # Training quality knobs
    top_k_words_for_ngrams: int = 10000  # Only use top-K words for n-gram extraction
    min_affix_productivity: int = 3  # Min distinct stems an affix must attach to
    word_priority_ratio: float = 0.7  # Fraction of root vocab to fill with full words first


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

        # Add common punctuation to root vocab so it can be modeled/decoded.
        # Keep this small and language-agnostic.
        for tok in [".", ",", "!", "?", ":", ";", "'", "\"", "(", ")", "[", "]", "{", "}"]:
            self.root_vocab.add_token(tok)
        
        # Add null affix for prefix/suffix vocabs
        self.prefix_vocab.add_token(NULL_AFFIX)
        self.suffix_vocab.add_token(NULL_AFFIX)
        
        # Learned patterns from training
        self._trained = False
        self._prefix_patterns: Set[str] = set()
        self._suffix_patterns: Set[str] = set()
        
        # Word-level parse cache for speed (word -> (prefix, root, suffix))
        self._parse_cache: Dict[str, Tuple[str, str, str]] = {}
        self._parse_cache_max_size = self.config.parse_cache_size
        
        # Special token IDs for easy access
        self.pad_token_id = self.root_vocab.get_id(PAD_TOKEN)
        self.unk_token_id = self.root_vocab.get_id(UNK_TOKEN)
        self.bos_token_id = self.root_vocab.get_id(BOS_TOKEN)
        self.eos_token_id = self.root_vocab.get_id(EOS_TOKEN)
        self.null_affix_id = self.prefix_vocab.get_id(NULL_AFFIX)
        
        # Word boundary regex (works across scripts)
        # Split on whitespace and separate punctuation from words
        self._word_re = re.compile(r'\S+|\s+')
        # Punctuation that should be separate tokens (no space before when decoding)
        self._punct_no_space_before = {',', '.', '!', '?', ':', ';', "'", '"', ')', ']', '}'}
        self._punct_no_space_after = {'(', '[', '{', '"', "'"}
        # Pattern to split punctuation from words
        self._punct_split_re = re.compile(r"([.,!?;:\"'()\[\]{}])")
    
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
        """
        Split text into words (unicode-aware).
        
        Separates punctuation into individual tokens so they can be
        handled independently during encoding/decoding.
        """
        # First split on whitespace
        raw_tokens = self._word_re.findall(text)
        
        result = []
        for token in raw_tokens:
            if not token.strip():
                continue
            
            # Split punctuation from the token
            # e.g., "Hello," -> ["Hello", ","]
            parts = self._punct_split_re.split(token)
            for part in parts:
                if part:  # Skip empty strings
                    result.append(part)
        
        return result
    
    def _train_root_vocab(self, word_freq: Counter, verbose: bool = True) -> None:
        """
        Train root vocabulary prioritizing full words over n-grams.
        
        Strategy:
        1. Fill word_priority_ratio of vocab with full words (by frequency)
        2. Fill remainder with n-grams (from top-K words only)
        3. Avoid single characters as roots
        """
        if verbose:
            print("   Training root vocabulary (word-priority mode)...")
        
        target_size = self.config.root_vocab_size
        word_slots = int(target_size * self.config.word_priority_ratio)
        
        # Step 1: Add frequent FULL WORDS first (these become the primary roots)
        words_added = 0
        for word, freq in word_freq.most_common():
            if len(self.root_vocab) >= target_size:
                break
            if words_added >= word_slots:
                break
            if freq >= self.config.min_freq and len(word) >= self.config.min_root_len:
                self.root_vocab.add_token(word)
                words_added += 1
        
        if verbose:
            print(f"      Added {words_added} full words to root vocab")
        
        # Step 2: Fill remaining slots with n-grams (for fallback/subword coverage)
        # Only collect from top-K frequent words to reduce memory/time
        top_k = self.config.top_k_words_for_ngrams
        top_words = word_freq.most_common(top_k)
        
        ngram_freq = Counter()
        for n in range(self.config.min_root_len, 8):  # min_root_len to 7 char n-grams
            for word, freq in top_words:
                if len(word) >= n:
                    for i in range(len(word) - n + 1):
                        ngram = word[i:i+n]
                        # Skip if it's already a full word in vocab
                        if ngram not in self.root_vocab.token_to_id:
                            ngram_freq[ngram] += freq
        
        # Add n-grams to fill remaining slots
        ngrams_added = 0
        for ngram, freq in ngram_freq.most_common():
            if len(self.root_vocab) >= target_size:
                break
            if freq >= self.config.min_freq:
                self.root_vocab.add_token(ngram)
                ngrams_added += 1
        
        if verbose:
            print(f"      Added {ngrams_added} n-grams to root vocab")
            print(f"      Final root vocab size: {len(self.root_vocab)}")
    
    def _train_affix_vocabs(self, word_freq: Counter, verbose: bool = True) -> None:
        """
        Train prefix and suffix vocabularies by productivity.
        
        Productivity = number of distinct stems an affix attaches to.
        Prefer affixes that, when stripped, yield known roots.
        """
        if verbose:
            print("   Training affix vocabularies (productivity mode)...")
        
        # Track which stems each affix attaches to
        prefix_stems: Dict[str, Set[str]] = defaultdict(set)
        suffix_stems: Dict[str, Set[str]] = defaultdict(set)
        
        # Also track frequency for tie-breaking
        prefix_freq = Counter()
        suffix_freq = Counter()
        
        root_vocab_set = self.root_vocab.token_to_id
        
        # Collect potential affixes and their stems
        for word, freq in word_freq.items():
            if len(word) < self.config.min_root_len + 1:
                continue
            
            # Extract prefixes
            for i in range(1, min(self.config.max_affix_len + 1, len(word) - self.config.min_root_len + 1)):
                prefix = word[:i]
                stem = word[i:]
                if len(stem) >= self.config.min_root_len:
                    prefix_stems[prefix].add(stem)
                    prefix_freq[prefix] += freq
            
            # Extract suffixes
            for i in range(1, min(self.config.max_affix_len + 1, len(word) - self.config.min_root_len + 1)):
                suffix = word[-i:]
                stem = word[:-i]
                if len(stem) >= self.config.min_root_len:
                    suffix_stems[suffix].add(stem)
                    suffix_freq[suffix] += freq
        
        # Score affixes by: productivity + bonus for stems in root vocab
        def score_affix(affix: str, stems: Set[str], freq: int, is_prefix: bool) -> float:
            productivity = len(stems)
            if productivity < self.config.min_affix_productivity:
                return 0  # Filter out low-productivity affixes
            
            # Bonus: how many stems are known roots
            known_stems = sum(1 for s in stems if s in root_vocab_set)
            
            # Score = productivity + 2*known_stems + 0.001*freq (freq as tie-breaker)
            return productivity + 2 * known_stems + 0.001 * freq
        
        # Score and sort prefixes
        prefix_scores = [
            (prefix, score_affix(prefix, prefix_stems[prefix], prefix_freq[prefix], True))
            for prefix in prefix_stems
        ]
        prefix_scores.sort(key=lambda x: -x[1])
        
        # Add top prefixes
        prefixes_added = 0
        for prefix, score in prefix_scores:
            if len(self.prefix_vocab) >= self.config.prefix_vocab_size:
                break
            if score > 0:
                self.prefix_vocab.add_token(prefix)
                self._prefix_patterns.add(prefix)
                prefixes_added += 1
        
        if verbose:
            print(f"      Added {prefixes_added} productive prefixes")
        
        # Score and sort suffixes
        suffix_scores = [
            (suffix, score_affix(suffix, suffix_stems[suffix], suffix_freq[suffix], False))
            for suffix in suffix_stems
        ]
        suffix_scores.sort(key=lambda x: -x[1])
        
        # Add top suffixes
        suffixes_added = 0
        for suffix, score in suffix_scores:
            if len(self.suffix_vocab) >= self.config.suffix_vocab_size:
                break
            if score > 0:
                self.suffix_vocab.add_token(suffix)
                self._suffix_patterns.add(suffix)
                suffixes_added += 1
        
        if verbose:
            print(f"      Added {suffixes_added} productive suffixes")
    
    def _parse_word_cached(self, word: str) -> Tuple[str, str, str]:
        """
        Parse a word with caching for repeated words.
        
        Uses LRU-like cache bounded by parse_cache_size.
        """
        # Check cache first
        if word in self._parse_cache:
            return self._parse_cache[word]
        
        # Parse and cache
        result = self._parse_word(word)
        
        # Add to cache if not full
        if self._parse_cache_max_size > 0:
            if len(self._parse_cache) >= self._parse_cache_max_size:
                # Simple eviction: clear half the cache when full
                # (LRU would be better but adds overhead)
                keys_to_remove = list(self._parse_cache.keys())[:self._parse_cache_max_size // 2]
                for k in keys_to_remove:
                    del self._parse_cache[k]
            self._parse_cache[word] = result
        
        return result
    
    def clear_parse_cache(self) -> None:
        """Clear the word parse cache."""
        self._parse_cache.clear()
    
    def _parse_word(self, word: str) -> Tuple[str, str, str]:
        """
        Parse a word into (prefix, root, suffix).
        
        Uses bounded affix-length search: O(max_affix_len^2) instead of O(|P|·|S|).
        Returns (prefix, root, suffix) where prefix/suffix may be NULL_AFFIX.
        """
        if not word or len(word) < self.config.min_root_len:
            return (NULL_AFFIX, word if word else UNK_TOKEN, NULL_AFFIX)
        
        best_parse = (NULL_AFFIX, word, NULL_AFFIX)
        best_score = 0
        
        prefix_vocab_set = self.prefix_vocab.token_to_id
        suffix_vocab_set = self.suffix_vocab.token_to_id
        root_vocab_set = self.root_vocab.token_to_id
        min_root = self.config.min_root_len
        max_affix = self.config.max_affix_len
        word_len = len(word)
        
        # Try prefix + root + suffix combinations (bounded by affix lengths)
        # prefix_len from 0 to max_affix_len, suffix_len from 0 to max_affix_len
        for prefix_len in range(0, min(max_affix + 1, word_len - min_root + 1)):
            if prefix_len > 0:
                prefix_candidate = word[:prefix_len]
                if prefix_candidate not in prefix_vocab_set:
                    continue
            else:
                prefix_candidate = NULL_AFFIX
            
            remaining_after_prefix = word_len - prefix_len
            
            for suffix_len in range(0, min(max_affix + 1, remaining_after_prefix - min_root + 1)):
                if suffix_len > 0:
                    suffix_candidate = word[word_len - suffix_len:]
                    if suffix_candidate not in suffix_vocab_set:
                        continue
                else:
                    suffix_candidate = NULL_AFFIX
                
                # Extract root
                root_start = prefix_len
                root_end = word_len - suffix_len
                root_candidate = word[root_start:root_end]
                
                if len(root_candidate) < min_root:
                    continue
                
                # Score:
                # - Strongly prefer roots already in vocab (stability)
                # - Prefer longer roots when root is unknown (avoid wal+king vs walk+ing)
                # - Slightly penalize affix length to avoid over-splitting
                score = 0
                if root_candidate in root_vocab_set:
                    score += 1000
                score += 10 * len(root_candidate)
                score -= (prefix_len + suffix_len)
                
                if score > best_score:
                    best_score = score
                    best_parse = (prefix_candidate, root_candidate, suffix_candidate)
        
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
            prefix, root, suffix = self._parse_word_cached(word)
            
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
        Decode token IDs back to text with punctuation-aware spacing.
        
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
        
        # Build list of words first
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
        
        # Join with punctuation-aware spacing
        if not words:
            return ""
        
        result = [words[0]]
        for word in words[1:]:
            # Check if this word is punctuation that shouldn't have space before it
            if word and word[0] in self._punct_no_space_before:
                result.append(word)
            # Check if previous word ends with punctuation that shouldn't have space after
            elif result and result[-1] and result[-1][-1] in self._punct_no_space_after:
                result.append(word)
            else:
                result.append(" " + word)
        
        return "".join(result)
    
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
