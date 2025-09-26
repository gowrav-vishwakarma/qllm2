#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Byte Pair Encoding (BPE) Tokenizer Implementation
Proper subword tokenization for better text generation
"""

import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
import os

class BPETokenizer:
    """Byte Pair Encoding tokenizer for better text processing"""
    
    def __init__(self, vocab_size=50257, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<eos>': 2,
            '<bos>': 3,
            '<sep>': 4,
        }
        
        # Vocabulary and merges
        self.vocab = self.special_tokens.copy()
        self.merges = []
        self.word_freqs = {}
        
        # Regex for tokenization (simplified for Python compatibility)
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")
        
        print(f"🔄 Initializing BPE tokenizer with target vocab size: {vocab_size}")
    
    def _get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        """Get word frequencies from training texts"""
        word_freqs = defaultdict(int)
        
        for text in texts:
            # Simple word tokenization
            words = text.split()
            for word in words:
                # Add BOS and EOS markers
                word_freqs[f'<bos>{word}<eos>'] += 1
        
        return dict(word_freqs)
    
    def _get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Get statistics of adjacent symbol pairs"""
        pairs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        
        return dict(pairs)
    
    def _merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        """Merge the most frequent pair in the vocabulary"""
        bigram = ' '.join(pair)
        new_word_freqs = {}
        
        for word in word_freqs:
            new_word = word.replace(' '.join(pair), bigram)
            new_word_freqs[new_word] = word_freqs[word]
        
        return new_word_freqs
    
    def train(self, texts: List[str], max_iterations: int = 10000):
        """Train the BPE tokenizer on the given texts"""
        print(f"📚 Training BPE tokenizer on {len(texts)} texts...")
        
        # Get word frequencies
        self.word_freqs = self._get_word_freqs(texts)
        print(f"📊 Found {len(self.word_freqs)} unique words")
        
        # Initialize vocabulary with characters
        vocab = set()
        for word in self.word_freqs.keys():
            for char in word:
                vocab.add(char)
        
        # Add characters to vocabulary
        for char in sorted(vocab):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        print(f"📊 Initial vocabulary size: {len(self.vocab)}")
        
        # BPE training
        num_merges = 0
        while len(self.vocab) < self.vocab_size and num_merges < max_iterations:
            pairs = self._get_stats(self.word_freqs)
            if not pairs:
                break
            
            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Skip if frequency is too low
            if pairs[best_pair] < self.min_frequency:
                break
            
            # Merge the pair
            self.word_freqs = self._merge_vocab(best_pair, self.word_freqs)
            self.merges.append(best_pair)
            
            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
            
            num_merges += 1
            
            if num_merges % 1000 == 0:
                print(f"📈 Merged {num_merges} pairs, vocab size: {len(self.vocab)}")
        
        print(f"✅ BPE training completed: {num_merges} merges, final vocab size: {len(self.vocab)}")
        
        # Create reverse vocabulary
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using BPE"""
        if word in self.vocab:
            return [word]
        
        # Start with characters
        tokens = list(word)
        
        # Apply merges
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(''.join(pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs"""
        # Simple word tokenization
        words = text.split()
        tokens = []
        
        # Add BOS token
        tokens.append(self.vocab['<bos>'])
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab['<unk>'])
        
        # Add EOS token
        tokens.append(self.vocab['<eos>'])
        
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            tokens[-1] = self.vocab['<eos>']  # Ensure EOS at end
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.idx_to_token:
                token = self.idx_to_token[token_id]
                if token not in ['<pad>', '<eos>', '<bos>', '<sep>']:
                    tokens.append(token)
        
        # Join tokens and clean up
        text = ''.join(tokens)
        
        # Add spaces between words (simple heuristic)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = data['merges']
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"📂 Tokenizer loaded from {filepath}")
    
    def __len__(self):
        return len(self.vocab)

def create_bpe_tokenizer_from_texts(texts: List[str], vocab_size: int = 50257) -> BPETokenizer:
    """Create and train a BPE tokenizer from texts"""
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    return tokenizer

# Test the tokenizer
if __name__ == "__main__":
    # Sample texts for testing
    sample_texts = [
        "The brain-inspired language model uses spiking neurons.",
        "Artificial intelligence is revolutionizing technology.",
        "Memory consolidation helps with learning and retention.",
        "Consciousness mechanisms enable self-awareness.",
        "Neural networks process information efficiently."
    ] * 100  # Repeat for more training data
    
    # Create and train tokenizer
    tokenizer = create_bpe_tokenizer_from_texts(sample_texts, vocab_size=1000)
    
    # Test encoding/decoding
    test_text = "The brain-inspired model is amazing!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {len(tokenizer)}")
