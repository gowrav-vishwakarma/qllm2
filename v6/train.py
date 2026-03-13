"""
V6 Training Script.

Usage:
    python -m v6.train --size small-matched --epochs 10 --max_samples 100000
    python -m v6.train --size tiny --epochs 2 --max_samples 100  # smoke test
    python -m v6.train --no_working_memory  # ablation without working memory
    python -m v6.train --resume checkpoints_v6/best_model.pt
"""

import json
import os
import re
import sys
import time
import math
import argparse
import warnings
import unicodedata
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

warnings.filterwarnings(
    'ignore',
    message=r'.*Online softmax is disabled.*Inductor.*split the reduction.*',
    category=UserWarning,
    module=r'torch\._inductor\.lowering',
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from v6.model import PhaseFieldLM, create_model, ModelOutput
from v6.config import V6Config, get_config
from v6.init import list_strategies
from v6.objectives import SpanCorruptionDataset, DelayedRecallDataset


class TeeLogger:
    """Writes to both stdout and a log file with timestamps."""

    def __init__(self, log_path: Path, mode: str = 'w'):
        self._stdout = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, mode, buffering=1)
        self._at_line_start = True

    def write(self, text: str):
        self._stdout.write(text)
        for char in text:
            if self._at_line_start and char != '\n':
                ts = datetime.now().strftime('[%Y-%m-%d %H:%M:%S] ')
                self._file.write(ts)
                self._at_line_start = False
            self._file.write(char)
            if char == '\n':
                self._at_line_start = True
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    @property
    def encoding(self):
        return self._stdout.encoding

    def fileno(self):
        return self._stdout.fileno()

    def isatty(self):
        return self._stdout.isatty()


class TextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int = 512):
        self.seq_len = seq_len
        n_chunks = len(tokens) // (seq_len + 1)
        self.data = tokens[:n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {'input_ids': chunk[:-1], 'labels': chunk[1:]}


_MOJIBAKE_REPLACEMENTS = [
    ('\u00e2\u0080\u009c', '\u201c'),  # left double quote
    ('\u00e2\u0080\u009d', '\u201d'),  # right double quote
    ('\u00e2\u0080\u0098', '\u2018'),  # left single quote
    ('\u00e2\u0080\u0099', '\u2019'),  # right single quote
    ('\u00e2\u0080\u0093', '\u2013'),  # en dash
    ('\u00e2\u0080\u0094', '\u2014'),  # em dash
    ('\u00e2\u0080\u00a6', '\u2026'),  # ellipsis
    ('\u00c2\u00a1', '\u00a1'),        # inverted exclamation
    ('\u00c2\u00bf', '\u00bf'),        # inverted question
]

_MOJIBAKE_RE = re.compile(
    r'[\u00c2-\u00c3][\u0080-\u00bf]|'
    r'\u00e2[\u0080-\u00bf][\u0080-\u00bf]'
)


def repair_text(text: str) -> str:
    """Fix common mojibake from UTF-8 bytes misinterpreted as Latin-1/CP1252."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = unicodedata.normalize('NFC', text)
    try:
        recovered = text.encode('cp1252').decode('utf-8')
        if not _MOJIBAKE_RE.search(recovered):
            return recovered
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    for bad, good in _MOJIBAKE_REPLACEMENTS:
        text = text.replace(bad, good)
    return text


def _tokenize_batch(texts: List[str], tokenizer, batch_size: int = 512) -> List[int]:
    """Batch-encode texts and flatten with EOS separators."""
    all_tokens = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)['input_ids']
        for ids in encoded:
            all_tokens.extend(ids)
            all_tokens.append(tokenizer.eos_token_id)
    return all_tokens


_CACHE_VERSION = 2


def load_tinystories(max_samples=20000, seq_len=512, text_repair=True,
                     use_cache=True, max_val_samples=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    repair_tag = "r1" if text_repair else "r0"
    cache_tag = f"v{_CACHE_VERSION}_ms{max_samples}_sl{seq_len}_{repair_tag}"

    def _process_split(split_name, limit):
        cache_path = Path(".cache") / "v6_tokens" / f"{split_name}_{cache_tag}.pt"

        if use_cache and cache_path.exists():
            cached = torch.load(cache_path, weights_only=False)
            tokens = cached['tokens']
            stats = cached.get('stats', {})
            print(f"[cache] Loaded {split_name} from {cache_path} "
                  f"({len(tokens):,} tokens)")
            return tokens, stats

        from datasets import load_dataset
        print(f"Loading TinyStories {split_name} (limit={limit})...")
        ds = load_dataset('roneneldan/TinyStories', split=split_name)
        texts = [item['text'] for item in ds if item['text'].strip()]
        if limit:
            texts = texts[:limit]

        stats = {'stories_raw': len(texts), 'repaired': 0, 'mojibake_before': 0,
                 'mojibake_after': 0}

        if text_repair:
            stats['mojibake_before'] = sum(
                1 for t in texts if _MOJIBAKE_RE.search(t)
            )
            repaired = []
            for t in texts:
                fixed = repair_text(t)
                if fixed != t:
                    stats['repaired'] += 1
                repaired.append(fixed)
            texts = repaired
            stats['mojibake_after'] = sum(
                1 for t in texts if _MOJIBAKE_RE.search(t)
            )
            print(f"  Text repair: {stats['repaired']} stories fixed, "
                  f"mojibake markers {stats['mojibake_before']} -> {stats['mojibake_after']}")

        print(f"  Tokenizing {len(texts)} {split_name} texts (batched)...")
        token_list = _tokenize_batch(texts, tokenizer)
        tokens = torch.tensor(token_list, dtype=torch.long)
        stats['tokens'] = len(tokens)
        print(f"  {split_name}: {stats['stories_raw']} stories, {len(tokens):,} tokens")

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'tokens': tokens, 'stats': stats,
                        'cache_version': _CACHE_VERSION}, cache_path)
            print(f"[cache] Saved {split_name} to {cache_path}")

        return tokens, stats

    try:
        train_tokens, train_stats = _process_split('train', max_samples)
        val_limit = max_val_samples or max(max_samples // 10, 1000) if max_samples else None
        val_tokens, val_stats = _process_split('validation', val_limit)
    except Exception as e:
        print(f"Failed to load TinyStories: {e}")
        print("Using random data as fallback.")
        return (_random_dataset(50257, seq_len, 1000),
                _random_dataset(50257, seq_len, 100), tokenizer)

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)
    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
    return train_ds, val_ds, tokenizer


def _random_dataset(vocab_size, seq_len, num_samples):
    tokens = torch.randint(1, vocab_size, (num_samples * (seq_len + 1),))
    return TextDataset(tokens, seq_len)


def load_wikitext103(max_samples=None, seq_len=512, use_cache=True, max_val_samples=None):
    """Load WikiText-103 for entity-rich, long-range dependency training.

    Uses HuggingFace wikitext/wikitext-103-raw-v1. Same tokenizer and
    TextDataset chunking as TinyStories. max_samples=None means use full dataset.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    limit_tag = f"ms{max_samples}" if max_samples else "full"
    cache_tag = f"v{_CACHE_VERSION}_{limit_tag}_sl{seq_len}"

    def _process_split(split_name, limit):
        cache_path = Path(".cache") / "v6_tokens" / f"wikitext103_{split_name}_{cache_tag}.pt"

        if use_cache and cache_path.exists():
            cached = torch.load(cache_path, weights_only=False)
            tokens = cached['tokens']
            stats = cached.get('stats', {})
            print(f"[cache] Loaded WikiText-103 {split_name} from {cache_path} "
                  f"({len(tokens):,} tokens)")
            return tokens, stats

        from datasets import load_dataset
        print(f"Loading WikiText-103 {split_name} (limit={limit})...")
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split_name)
        lines = [item['text'] for item in ds]

        if limit:
            lines = lines[:limit]

        stats = {'lines': len(lines)}

        # Tokenize in chunks to avoid memory issues with very long strings
        all_tokens = []
        chunk_size = 50000  # lines per batch
        for start in range(0, len(lines), chunk_size):
            chunk_lines = lines[start:start + chunk_size]
            chunk_text = '\n'.join(chunk_lines)
            ids = tokenizer.encode(chunk_text, add_special_tokens=False)
            all_tokens.extend(ids)

        tokens = torch.tensor(all_tokens, dtype=torch.long)
        stats['tokens'] = len(tokens)
        print(f"  {split_name}: {stats['lines']} lines, {len(tokens):,} tokens")

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'tokens': tokens, 'stats': stats,
                        'cache_version': _CACHE_VERSION}, cache_path)
            print(f"[cache] Saved {split_name} to {cache_path}")

        return tokens, stats

    try:
        train_tokens, train_stats = _process_split('train', max_samples)
        val_limit = max_val_samples or (max(max_samples // 10, 1000) if max_samples else None)
        val_tokens, val_stats = _process_split('validation', val_limit)
    except Exception as e:
        print(f"Failed to load WikiText-103: {e}")
        print("Using random data as fallback.")
        return (_random_dataset(50257, seq_len, 1000),
                _random_dataset(50257, seq_len, 100), tokenizer)

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)
    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
    return train_ds, val_ds, tokenizer


def load_pg19(max_samples=None, seq_len=1024, use_cache=True, max_val_samples=None):
    """Load PG-19 (Project Gutenberg) for long-range narrative training.

    Full-length books with characters persisting for thousands of tokens.
    Tests multi-timescale SSM slow lanes and working memory over long contexts.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    limit_tag = f"ms{max_samples}" if max_samples else "full"
    cache_tag = f"v{_CACHE_VERSION}_{limit_tag}_sl{seq_len}"

    def _process_split(split_name, limit):
        cache_path = Path(".cache") / "v6_tokens" / f"pg19_{split_name}_{cache_tag}.pt"

        if use_cache and cache_path.exists():
            cached = torch.load(cache_path, weights_only=False)
            tokens = cached['tokens']
            stats = cached.get('stats', {})
            print(f"[cache] Loaded PG-19 {split_name} from {cache_path} "
                  f"({len(tokens):,} tokens)")
            return tokens, stats

        from datasets import load_dataset
        print(f"Loading PG-19 {split_name} (limit={limit})...")
        ds = load_dataset('pg19', split=split_name)
        texts = [item['text'] for item in ds if item['text'].strip()]

        if limit:
            texts = texts[:limit]

        stats = {'books': len(texts)}
        print(f"  {split_name}: {stats['books']} books, tokenizing...")

        all_tokens = []
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(ids)
            all_tokens.append(tokenizer.eos_token_id)
            if (i + 1) % 500 == 0:
                print(f"    tokenized {i+1}/{len(texts)} books ({len(all_tokens):,} tokens so far)")

        tokens = torch.tensor(all_tokens, dtype=torch.long)
        stats['tokens'] = len(tokens)
        print(f"  {split_name}: {stats['books']} books, {len(tokens):,} tokens")

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'tokens': tokens, 'stats': stats,
                        'cache_version': _CACHE_VERSION}, cache_path)
            print(f"[cache] Saved {split_name} to {cache_path}")

        return tokens, stats

    try:
        train_tokens, train_stats = _process_split('train', max_samples)
        val_limit = max_val_samples or (max(max_samples // 10, 100) if max_samples else None)
        val_tokens, val_stats = _process_split('validation', val_limit)
    except Exception as e:
        print(f"Failed to load PG-19: {e}")
        print("Using random data as fallback.")
        return (_random_dataset(50257, seq_len, 1000),
                _random_dataset(50257, seq_len, 100), tokenizer)

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)
    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
    return train_ds, val_ds, tokenizer


class MixedDataset(Dataset):
    """Interleaved sampling from multiple TextDatasets with configurable ratios."""

    def __init__(self, datasets: list, weights: list):
        assert len(datasets) == len(weights)
        self.datasets = datasets
        total_w = sum(weights)
        self.probs = [w / total_w for w in weights]
        self.cumulative = []
        running = 0.0
        for p in self.probs:
            running += p
            self.cumulative.append(running)
        self._total_len = sum(len(d) for d in datasets)

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        import random
        r = random.random()
        for i, threshold in enumerate(self.cumulative):
            if r < threshold:
                ds = self.datasets[i]
                return ds[idx % len(ds)]
        return self.datasets[-1][idx % len(self.datasets[-1])]


def _load_wikipedia_tokens(tokenizer, max_samples=None, use_cache=True, seq_len=512):
    """Load Wikipedia (20220301.en) and return raw token tensor."""
    limit_tag = f"ms{max_samples}" if max_samples else "full"
    cache_tag = f"v{_CACHE_VERSION}_{limit_tag}_sl{seq_len}"
    cache_path = Path(".cache") / "v6_tokens" / f"wikipedia_train_{cache_tag}.pt"

    if use_cache and cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        tokens = cached['tokens']
        print(f"[cache] Loaded Wikipedia from {cache_path} ({len(tokens):,} tokens)")
        return tokens

    from datasets import load_dataset
    print(f"Loading Wikipedia (limit={max_samples})...")
    ds = load_dataset('wikipedia', '20220301.en', split='train')
    texts = [item['text'] for item in ds if item['text'].strip()]
    if max_samples:
        texts = texts[:max_samples]
    print(f"  Tokenizing {len(texts)} articles...")
    token_list = _tokenize_batch(texts, tokenizer)
    tokens = torch.tensor(token_list, dtype=torch.long)
    print(f"  Wikipedia: {len(texts)} articles, {len(tokens):,} tokens")

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'tokens': tokens, 'cache_version': _CACHE_VERSION}, cache_path)
        print(f"[cache] Saved to {cache_path}")

    return tokens


def _load_openwebtext_tokens(tokenizer, max_samples=None, use_cache=True, seq_len=512):
    """Load OpenWebText and return raw token tensor."""
    limit_tag = f"ms{max_samples}" if max_samples else "full"
    cache_tag = f"v{_CACHE_VERSION}_{limit_tag}_sl{seq_len}"
    cache_path = Path(".cache") / "v6_tokens" / f"openwebtext_train_{cache_tag}.pt"

    if use_cache and cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        tokens = cached['tokens']
        print(f"[cache] Loaded OpenWebText from {cache_path} ({len(tokens):,} tokens)")
        return tokens

    from datasets import load_dataset
    print(f"Loading OpenWebText (limit={max_samples})...")
    ds = load_dataset('openwebtext', split='train')
    texts = [item['text'] for item in ds if item['text'].strip()]
    if max_samples:
        texts = texts[:max_samples]
    print(f"  Tokenizing {len(texts)} documents...")
    token_list = _tokenize_batch(texts, tokenizer)
    tokens = torch.tensor(token_list, dtype=torch.long)
    print(f"  OpenWebText: {len(texts)} documents, {len(tokens):,} tokens")

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'tokens': tokens, 'cache_version': _CACHE_VERSION}, cache_path)
        print(f"[cache] Saved to {cache_path}")

    return tokens


def load_mixed(mix_spec: str, seq_len=512, max_tokens_per_source=None,
               use_cache=True):
    """Load a mixed corpus with configurable ratios.

    mix_spec format: "source1:weight1,source2:weight2,..."
    Example: "wikipedia:40,pg19:30,openwebtext:30"

    Available sources: wikipedia, pg19, wikitext103, openwebtext
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    source_loaders = {
        'wikipedia': lambda: _load_wikipedia_tokens(
            tokenizer, max_samples=max_tokens_per_source, use_cache=use_cache, seq_len=seq_len),
        'openwebtext': lambda: _load_openwebtext_tokens(
            tokenizer, max_samples=max_tokens_per_source, use_cache=use_cache, seq_len=seq_len),
    }

    parts = [p.strip() for p in mix_spec.split(',')]
    datasets_train = []
    weights = []

    for part in parts:
        name, weight_str = part.split(':')
        name = name.strip()
        weight = float(weight_str.strip())

        if name in source_loaders:
            tokens = source_loaders[name]()
        elif name == 'wikitext103':
            train_ds, _, _ = load_wikitext103(
                max_samples=max_tokens_per_source, seq_len=seq_len, use_cache=use_cache)
            datasets_train.append(train_ds)
            weights.append(weight)
            continue
        elif name == 'pg19':
            train_ds, _, _ = load_pg19(
                max_samples=max_tokens_per_source, seq_len=seq_len, use_cache=use_cache)
            datasets_train.append(train_ds)
            weights.append(weight)
            continue
        else:
            raise ValueError(f"Unknown source: {name}. Available: wikipedia, pg19, wikitext103, openwebtext")

        train_ds = TextDataset(tokens, seq_len)
        datasets_train.append(train_ds)
        weights.append(weight)

    print(f"\nMixed corpus:")
    for i, (part, ds) in enumerate(zip(parts, datasets_train)):
        print(f"  {part} -> {len(ds)} chunks (weight {weights[i]:.0f})")

    mixed_train = MixedDataset(datasets_train, weights)
    print(f"Total train chunks: {len(mixed_train)}")

    # Use WikiText-103 validation as the standard validation set for mixed training
    _, val_ds, _ = load_wikitext103(seq_len=seq_len, use_cache=use_cache)
    print(f"Validation (WikiText-103): {len(val_ds)} chunks")

    return mixed_train, val_ds, tokenizer


class ImageDataset(Dataset):
    """Wraps PIL images with torchvision transforms for diffusion training."""

    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img = self.images[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return {'image': self.transform(img)}


def load_image_dataset(config):
    """Load image dataset via HuggingFace datasets + torchvision transforms."""
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision required for image mode. Run: uv add torchvision")

    from datasets import load_dataset

    transform_train = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    if config.image_dataset == 'tiny_imagenet':
        print("Loading Tiny ImageNet...")
        ds = load_dataset('zh-plus/tiny-imagenet', split='train')
        images = [item['image'] for item in ds]
    elif config.image_dataset == 'cifar10':
        print("Loading CIFAR-10...")
        ds = load_dataset('cifar10', split='train')
        images = [item['img'] for item in ds]
    else:
        raise ValueError(f"Unknown image dataset: {config.image_dataset}")

    print(f"Loaded {len(images)} images")

    split = int(len(images) * 0.9)
    train_ds = ImageDataset(images[:split], transform_train)
    val_ds = ImageDataset(images[split:], transform_val)
    print(f"Image train: {len(train_ds)}, val: {len(val_ds)}, size: {config.image_size}x{config.image_size}")
    return train_ds, val_ds


def _resolve_amp_dtype(amp_dtype_str: str) -> Optional[torch.dtype]:
    """Resolve AMP dtype string to a torch dtype. Returns None if AMP should be disabled."""
    if not torch.cuda.is_available():
        return None
    if amp_dtype_str == 'bf16':
        return torch.bfloat16
    if amp_dtype_str == 'fp16':
        return torch.float16
    # 'auto': prefer bf16 on capable hardware
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


_NO_DECAY_SUFFIXES = {
    'log_A_real', 'log_A_imag', 'dt_bias',
}


def _build_param_groups(model: nn.Module, weight_decay: float):
    """Split parameters into decay / no-decay groups for AdamW.

    Decay group: 2-D weight matrices from nn.Linear, nn.Embedding, and
    ComplexLinear (weight_real, weight_imag) -- standard L2 regularization.

    No-decay group: everything else -- SSM eigenvalue params (log_A_real,
    log_A_imag), dt_bias, all biases, normalization scales, scalar gates,
    phase_rotations, 1-D parameters, and any explicitly listed names.
    """
    decay_params = []
    no_decay_params = []
    decay_names = []
    no_decay_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        suffix = name.split('.')[-1]

        if suffix in _NO_DECAY_SUFFIXES:
            no_decay_params.append(param)
            no_decay_names.append(name)
        elif suffix in ('bias', 'bias_real', 'bias_imag'):
            no_decay_params.append(param)
            no_decay_names.append(name)
        elif param.dim() >= 2 and suffix in ('weight', 'weight_real', 'weight_imag'):
            decay_params.append(param)
            decay_names.append(name)
        else:
            no_decay_params.append(param)
            no_decay_names.append(name)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Param groups: {len(decay_params)} tensors ({n_decay:,} params) with weight decay, "
          f"{len(no_decay_params)} tensors ({n_no_decay:,} params) without")

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]


def _build_lr_scheduler(optimizer, schedule: str, warmup_steps: int, total_steps: int):
    """Build LR scheduler: plain cosine or linear-warmup + cosine decay."""
    total_steps = max(total_steps, 1)
    if schedule == 'warmup_cosine' and warmup_steps > 0:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


def compute_text_quality(text: str) -> Dict[str, float]:
    """Behavioral quality metrics beyond perplexity.

    Returns dict with:
      - repeat_3gram: fraction of 3-grams that are repeated
      - repeat_4gram: fraction of 4-grams that are repeated
      - restart_frag: number of mid-text restart patterns detected
      - unique_word_ratio: unique words / total words
    """
    words = text.split()
    n_words = max(len(words), 1)

    def ngram_repeat_rate(tokens, n):
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return 1.0 - len(set(ngrams)) / max(len(ngrams), 1)

    restart_patterns = [
        'once upon a time',
        '<|endoftext|>',
    ]
    lower_text = text.lower()
    restart_count = 0
    for pat in restart_patterns:
        occurrences = lower_text.count(pat)
        if occurrences > 1:
            restart_count += occurrences - 1

    return {
        'repeat_3gram': ngram_repeat_rate(words, 3),
        'repeat_4gram': ngram_repeat_rate(words, 4),
        'restart_frag': float(restart_count),
        'unique_word_ratio': len(set(words)) / n_words,
    }


def _notify_discord(content: str) -> None:
    """Optionally send a copy of the message to Discord if DISCORD_HOOK is set.
    Does not replace or suppress any existing logging; console/file logging is unchanged."""
    hook = os.environ.get("DISCORD_HOOK", "").strip()
    if not hook:
        return
    if len(content) > 2000:
        content = content[:1997] + "..."
    try:
        payload = json.dumps({"content": content}).encode("utf-8")
        req = urllib.request.Request(
            hook,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "qllm2-discord-notify/1.0",
            },
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[Discord] Webhook send failed: {e}", file=sys.stderr)


class Trainer:
    def __init__(
        self,
        model: PhaseFieldLM,
        config: V6Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer=None,
        checkpoint_dir: str = 'checkpoints_v6',
        start_epoch: int = 0,
        save_checkpoints: bool = True,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = start_epoch
        self.save_checkpoints = save_checkpoints
        self.verbose = verbose
        self.gen_every = 0
        self.gen_prompt = "The"
        self.log_interval = 50

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        param_groups = _build_param_groups(model, config.weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
        )

        total_steps = config.max_epochs * len(train_loader)
        self.scheduler = _build_lr_scheduler(
            self.optimizer, config.lr_schedule, config.warmup_steps, total_steps,
        )

        self.amp_dtype = _resolve_amp_dtype(config.amp_dtype)
        self.use_amp = self.amp_dtype is not None
        # GradScaler only needed for float16; bfloat16 doesn't need loss scaling
        self.scaler = (torch.amp.GradScaler('cuda')
                       if self.use_amp and self.amp_dtype == torch.float16
                       else None)
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_ppl = float('inf')

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_div_loss = 0.0
        total_raw_div = 0.0
        num_batches = 0
        epoch_start = time.time()
        first_step_start = None
        log_interval_start = epoch_start
        log_interval_tokens = 0
        total_epoch_tokens = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx == 0:
                first_step_start = time.time()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            seq_len = input_ids.shape[1]
            batch_tokens = input_ids.shape[0] * seq_len

            loss_mask = batch.get('loss_mask')
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp,
                                    dtype=self.amp_dtype or torch.float16):
                output = self.model(input_ids)

                logits = output.logits.view(-1, output.logits.size(-1))
                if loss_mask is not None:
                    flat_mask = loss_mask.view(-1).float()
                    per_token_loss = F.cross_entropy(
                        logits, labels.view(-1), reduction='none',
                    )
                    ce_loss = (per_token_loss * flat_mask).sum() / flat_mask.sum().clamp(min=1)
                else:
                    ce_loss = F.cross_entropy(logits, labels.view(-1))

                loss = ce_loss
                div_loss_val = 0.0
                raw_div_val = 0.0
                if output.diversity_loss is not None:
                    raw_div_val = output.diversity_loss.item()
                    total_steps = self.config.max_epochs * len(self.train_loader)
                    progress = min(self.global_step / max(total_steps, 1), 1.0)
                    div_w = self.config.diversity_loss_weight + (
                        self.config.diversity_loss_floor - self.config.diversity_loss_weight
                    ) * progress
                    div_w = max(div_w, self.config.diversity_loss_floor)
                    div_loss = output.diversity_loss * div_w
                    loss = loss + div_loss
                    div_loss_val = div_loss.item()

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_div_loss += div_loss_val
            total_raw_div += raw_div_val
            num_batches += 1
            total_epoch_tokens += batch_tokens
            log_interval_tokens += batch_tokens

            if self.verbose and batch_idx == 0 and first_step_start is not None:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                first_step_s = time.time() - first_step_start
                print(f"  First step wall time: {first_step_s:.1f}s")

            if self.verbose and batch_idx % self.log_interval == 0:
                ppl = math.exp(min(ce_loss.item(), 20))
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                avg_tok_s = total_epoch_tokens / elapsed if elapsed > 0 else 0
                interval_elapsed = time.time() - log_interval_start
                inst_tok_s = log_interval_tokens / interval_elapsed if interval_elapsed > 0 else 0

                n_total = len(self.train_loader)
                pct = 100.0 * (batch_idx + 1) / n_total
                remaining = elapsed / (batch_idx + 1) * (n_total - batch_idx - 1) if batch_idx > 0 else 0
                eta_m, eta_s = divmod(int(remaining), 60)

                line = (
                    f"  [{epoch+1}] {batch_idx}/{n_total} ({pct:.0f}%) "
                    f"loss={ce_loss.item():.4f} ppl={ppl:.1f} "
                    f"div={raw_div_val:.2e} wdiv={div_loss_val:.2e} lr={lr:.2e} "
                    f"| {inst_tok_s:.0f} tok/s (avg {avg_tok_s:.0f}) "
                    f"ETA {eta_m}m{eta_s:02d}s"
                )
                if self.device.type == 'cuda':
                    mem = torch.cuda.memory_allocated() / 1e9
                    mem_res = torch.cuda.max_memory_allocated() / 1e9
                    line += f" | GPU {mem:.1f}/{mem_res:.1f}GB"
                print(line)

                log_interval_start = time.time()
                log_interval_tokens = 0

            if (self.gen_every > 0 and batch_idx > 0
                    and batch_idx % self.gen_every == 0
                    and self.tokenizer is not None):
                try:
                    text = self.generate_sample(self.gen_prompt)
                    print(f"  [mid-epoch sample @ batch {batch_idx}]")
                    print(f"  Prompt: {self.gen_prompt}")
                    print(f"  Generated: {text}")
                    ppl = math.exp(min(ce_loss.item(), 20))
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - epoch_start
                    avg_tok_s = total_epoch_tokens / elapsed if elapsed > 0 else 0
                    _gen_msg = (
                        f"**[gen_every]** Epoch {epoch+1} batch {batch_idx}\n"
                        f"loss={ce_loss.item():.4f} ppl={ppl:.1f} div={raw_div_val:.2e} wdiv={div_loss_val:.2e} lr={lr:.2e} | {avg_tok_s:.0f} tok/s\n"
                        f"Prompt: {self.gen_prompt}\n"
                        f"Generated: {(text[:800] + '...') if len(text) > 800 else text}"
                    )
                    _notify_discord(_gen_msg)
                except Exception:
                    pass
                self.model.train()

        epoch_elapsed = time.time() - epoch_start
        avg_tok_s = total_epoch_tokens / epoch_elapsed if epoch_elapsed > 0 else 0
        return {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'div_loss': total_div_loss / num_batches,
            'raw_div': total_raw_div / num_batches,
            'ppl': math.exp(min(total_ce_loss / num_batches, 20)),
            'avg_tok_s': avg_tok_s,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            loss_mask = batch.get('loss_mask')
            output = self.model(input_ids)
            logits = output.logits.view(-1, output.logits.size(-1))
            if loss_mask is not None:
                flat_mask = loss_mask.to(self.device).view(-1).float()
                per_token = F.cross_entropy(logits, labels.view(-1), reduction='none')
                loss = (per_token * flat_mask).sum() / flat_mask.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(logits, labels.view(-1))
            total_loss += loss.item()
            num_batches += 1
        if num_batches == 0:
            return {}
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss, 'val_ppl': math.exp(min(avg_loss, 20))}

    @torch.no_grad()
    def generate_sample(self, prompt="The quick brown", max_tokens=100):
        self.model.eval()
        model_to_gen = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        generated = model_to_gen.generate(
            prompt_tensor, max_new_tokens=max_tokens, temperature=0.8,
            top_k=50, top_p=0.9, repetition_penalty=1.2,
        )
        return self.tokenizer.decode(generated[0].tolist())

    def save_checkpoint(self, name: str):
        path = self.checkpoint_dir / name
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        ckpt = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'best_val_ppl': self.best_val_ppl,
            'epoch': self._current_epoch,
            'config': self.config.to_dict(),
        }
        if hasattr(self.model, 'initializer_info'):
            ckpt['init_strategy'] = self.model.initializer_info['init_strategy']
            ckpt['init_seed'] = self.model.initializer_info['init_seed']
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")

    def train(self):
        training_start = time.time()
        print(f"\nTraining on {self.device}")
        params = self.model.count_parameters() if hasattr(self.model, 'count_parameters') else {}
        if params:
            print(f"Parameters: {params}")
            print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")
        print(f"Epochs: {self.start_epoch+1}..{self.config.max_epochs}, Batches/epoch: {len(self.train_loader)}")
        print()

        _start_msg = (
            f"**Training started** (AR)\n"
            f"Device: {self.device}\n"
            f"Epochs: {self.start_epoch+1}..{self.config.max_epochs} | Batches/epoch: {len(self.train_loader)}"
        )
        if params:
            _start_msg += f"\nParams: {params.get('total', 0):,} ({params.get('total', 0)/1e6:.1f}M)"
        _notify_discord(_start_msg)

        for epoch in range(self.start_epoch, self.config.max_epochs):
            self._current_epoch = epoch
            val_metrics = None
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.max_epochs}")
            print('=' * 60)

            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            line = (
                f"Epoch {epoch+1}/{self.config.max_epochs} | "
                f"Train Loss: {train_metrics['ce_loss']:.4f} "
                f"PPL: {train_metrics['ppl']:.2f} "
                f"div={train_metrics['raw_div']:.2e} wdiv={train_metrics['div_loss']:.2e} | "
                f"{train_metrics['avg_tok_s']:.0f} tok/s | "
                f"Time: {epoch_time:.1f}s"
            )

            is_best = False
            if self.val_loader is not None and len(self.val_loader) > 0:
                val_metrics = self.validate()
                line += (
                    f" | Val Loss: {val_metrics['val_loss']:.4f} "
                    f"PPL: {val_metrics['val_ppl']:.2f}"
                )
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.best_val_ppl = val_metrics['val_ppl']
                    line += " *best*"
                    is_best = True
            print(line)

            if self.save_checkpoints and is_best:
                self.save_checkpoint('best_model.pt')
            if self.save_checkpoints and (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

            epoch_text = ""
            if self.tokenizer is not None:
                try:
                    text = self.generate_sample(self.gen_prompt)
                    epoch_text = text
                    print(f"\nPrompt: {self.gen_prompt}")
                    print(f"Generated: {text}")
                    qm = compute_text_quality(text)
                    print(
                        f"  Quality: rep3={qm['repeat_3gram']:.3f} "
                        f"rep4={qm['repeat_4gram']:.3f} "
                        f"restarts={qm['restart_frag']:.0f} "
                        f"uniq={qm['unique_word_ratio']:.3f}"
                    )
                except Exception as e:
                    print(f"(Sample generation failed: {e})")
            _ep_msg = (
                f"**Epoch {epoch+1}/{self.config.max_epochs}**\n"
                f"Train Loss: {train_metrics['ce_loss']:.4f} PPL: {train_metrics['ppl']:.2f} "
                f"div={train_metrics['raw_div']:.2e} wdiv={train_metrics['div_loss']:.2e} | "
                f"{train_metrics['avg_tok_s']:.0f} tok/s | Time: {epoch_time:.1f}s"
            )
            if val_metrics is not None:
                _ep_msg += f"\nVal Loss: {val_metrics['val_loss']:.4f} PPL: {val_metrics['val_ppl']:.2f}"
            if epoch_text:
                _ep_msg += f"\nPrompt: {self.gen_prompt}\nGenerated: {(epoch_text[:600] + '...') if len(epoch_text) > 600 else epoch_text}"
            _notify_discord(_ep_msg)

        self._current_epoch = self.config.max_epochs - 1
        if self.save_checkpoints:
            self.save_checkpoint('final_model.pt')

        total_time = time.time() - training_start
        print(f"\nTraining complete!")
        print(f"Total wall time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        print(f"Best Val Loss: {self.best_val_loss:.4f}, Best Val PPL: {self.best_val_ppl:.2f}")


class DiffusionTrainer:
    """Training loop for diffusion modes (text and image). Mirrors Trainer API."""

    def __init__(
        self,
        model,
        config: V6Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer=None,
        checkpoint_dir: str = 'checkpoints_v6',
        start_epoch: int = 0,
        save_checkpoints: bool = True,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = start_epoch
        self.save_checkpoints = save_checkpoints
        self.verbose = verbose
        self.gen_every = 0
        self.sample_dir = None
        self.log_interval = 50

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        param_groups = _build_param_groups(model, config.weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
        )

        total_steps = config.max_epochs * len(train_loader)
        self.scheduler = _build_lr_scheduler(
            self.optimizer, config.lr_schedule, config.warmup_steps, total_steps,
        )

        self.amp_dtype = _resolve_amp_dtype(config.amp_dtype)
        self.use_amp = self.amp_dtype is not None
        self.scaler = (torch.amp.GradScaler('cuda')
                       if self.use_amp and self.amp_dtype == torch.float16
                       else None)
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _get_input(self, batch):
        if self.config.mode == 'diffusion_text':
            return batch['input_ids'].to(self.device, non_blocking=True)
        else:
            return batch['image'].to(self.device, non_blocking=True)

    def _diversity_weight(self):
        total_steps = self.config.max_epochs * len(self.train_loader)
        progress = min(self.global_step / max(total_steps, 1), 1.0)
        div_w = self.config.diversity_loss_weight + (
            self.config.diversity_loss_floor - self.config.diversity_loss_weight
        ) * progress
        return max(div_w, self.config.diversity_loss_floor)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_diff_loss = 0.0
        total_div_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            x = self._get_input(batch)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp,
                                    dtype=self.amp_dtype or torch.float16):
                output = self.model(x)
                loss = output.loss
                div_loss_val = 0.0
                if output.diversity_loss is not None:
                    div_w = self._diversity_weight()
                    div_loss = output.diversity_loss * div_w
                    loss = loss + div_loss
                    div_loss_val = div_loss.item()

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1
            total_loss += loss.item()
            total_diff_loss += output.loss.item()
            total_div_loss += div_loss_val
            num_batches += 1

            if self.verbose and batch_idx % self.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                samples_per_sec = (batch_idx + 1) * self.config.batch_size / elapsed if elapsed > 0 else 0
                n_total = len(self.train_loader)
                pct = 100.0 * (batch_idx + 1) / n_total
                remaining = elapsed / (batch_idx + 1) * (n_total - batch_idx - 1) if batch_idx > 0 else 0
                eta_m, eta_s = divmod(int(remaining), 60)
                print(
                    f"  [{epoch+1}] {batch_idx}/{n_total} ({pct:.0f}%) "
                    f"diff_loss={output.loss.item():.4f} "
                    f"div={div_loss_val:.2e} lr={lr:.2e} "
                    f"| {samples_per_sec:.1f} samples/s "
                    f"ETA {eta_m}m{eta_s:02d}s"
                )

            if (self.gen_every > 0 and batch_idx > 0
                    and batch_idx % self.gen_every == 0):
                gen_text = self._generate_samples(epoch, batch_idx)
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                samples_per_sec = (batch_idx + 1) * self.config.batch_size / elapsed if elapsed > 0 else 0
                _gen_msg = (
                    f"**[gen_every]** Diffusion Epoch {epoch+1} batch {batch_idx}\n"
                    f"diff_loss={output.loss.item():.4f} div={div_loss_val:.2e} lr={lr:.2e} | {samples_per_sec:.1f} samples/s\n"
                )
                if gen_text:
                    _gen_msg += f"Generated: {(gen_text[:800] + '...') if len(gen_text) > 800 else gen_text}"
                _notify_discord(_gen_msg)
                self.model.train()

        return {
            'loss': total_loss / max(num_batches, 1),
            'diff_loss': total_diff_loss / max(num_batches, 1),
            'div_loss': total_div_loss / max(num_batches, 1),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_loader:
            x = self._get_input(batch)
            output = self.model(x)
            total_loss += output.loss.item()
            num_batches += 1
        if num_batches == 0:
            return {}
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}

    def _generate_samples(self, epoch: int, batch_idx: int = 0) -> str:
        """Generate and display/save samples. Returns a string for Discord (decoded text or save path)."""
        self.model.eval()
        model_to_sample = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        out_text = ""

        if self.config.mode == 'diffusion_text' and self.tokenizer is not None:
            seq_len = self.config.max_seq_len
            tokens = model_to_sample.sample(
                batch_size=2, seq_len=seq_len, device=self.device,
                num_steps=min(50, self.config.diffusion_steps),
            )
            parts = []
            for i in range(tokens.shape[0]):
                text = self.tokenizer.decode(tokens[i].tolist(), skip_special_tokens=True)
                parts.append(text)
                print(f"  [sample {i+1}] {text[:200]}")
            out_text = "\n---\n".join(parts) if parts else ""

        elif self.config.mode == 'diffusion_image' and self.sample_dir is not None:
            try:
                from torchvision.utils import save_image
                seq_len = (self.config.image_size // self.config.patch_size) ** 2
                images = model_to_sample.sample(
                    batch_size=16, seq_len=seq_len, device=self.device,
                    num_steps=min(50, self.config.diffusion_steps),
                )
                if images.dim() == 4:
                    images = (images + 1) / 2  # [-1,1] -> [0,1]
                    path = self.sample_dir / f"samples_e{epoch}_b{batch_idx}.png"
                    save_image(images, path, nrow=4)
                    out_text = f"Samples saved to {path}"
                    print(f"  [samples saved to {path}]")
            except Exception as e:
                print(f"  (sample generation failed: {e})")

        return out_text

    def save_checkpoint(self, name: str):
        path = self.checkpoint_dir / name
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        ckpt = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'epoch': self._current_epoch,
            'config': self.config.to_dict(),
        }
        if hasattr(self.model, 'initializer_info'):
            ckpt['init_strategy'] = self.model.initializer_info['init_strategy']
            ckpt['init_seed'] = self.model.initializer_info['init_seed']
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")

    def train(self):
        training_start = time.time()
        print(f"\nTraining on {self.device} (mode: {self.config.mode})")
        params = self.model.count_parameters() if hasattr(self.model, 'count_parameters') else {}
        if params:
            print(f"Parameters: {params}")
            print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")
        print(f"Diffusion steps: {self.config.diffusion_steps}, schedule: {self.config.noise_schedule}")
        print(f"Prediction target: {self.config.prediction_target}")
        print(f"Epochs: {self.start_epoch+1}..{self.config.max_epochs}, Batches/epoch: {len(self.train_loader)}")
        print()

        _start_msg = (
            f"**Training started** (Diffusion)\n"
            f"Device: {self.device} | mode: {self.config.mode}\n"
            f"Steps: {self.config.diffusion_steps} | schedule: {self.config.noise_schedule} | target: {self.config.prediction_target}\n"
            f"Epochs: {self.start_epoch+1}..{self.config.max_epochs} | Batches/epoch: {len(self.train_loader)}"
        )
        if params:
            _start_msg += f"\nParams: {params.get('total', 0):,} ({params.get('total', 0)/1e6:.1f}M)"
        _notify_discord(_start_msg)

        for epoch in range(self.start_epoch, self.config.max_epochs):
            self._current_epoch = epoch
            val_metrics = None
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.max_epochs}")
            print('=' * 60)

            epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            line = (
                f"Epoch {epoch+1}/{self.config.max_epochs} | "
                f"Diff Loss: {train_metrics['diff_loss']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            is_best = False
            if self.val_loader is not None and len(self.val_loader) > 0:
                val_metrics = self.validate()
                if val_metrics:
                    line += f" | Val Loss: {val_metrics['val_loss']:.4f}"
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        line += " *best*"
                        is_best = True
            print(line)

            if self.save_checkpoints and is_best:
                self.save_checkpoint('best_model.pt')
            if self.save_checkpoints and (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

            epoch_gen = self._generate_samples(epoch)
            _ep_msg = (
                f"**Epoch {epoch+1}/{self.config.max_epochs}** (Diffusion)\n"
                f"Diff Loss: {train_metrics['diff_loss']:.4f} | Time: {epoch_time:.1f}s"
            )
            if val_metrics is not None:
                _ep_msg += f"\nVal Loss: {val_metrics['val_loss']:.4f}"
            if epoch_gen:
                _ep_msg += f"\nGenerated: {(epoch_gen[:600] + '...') if len(epoch_gen) > 600 else epoch_gen}"
            _notify_discord(_ep_msg)

        self._current_epoch = self.config.max_epochs - 1
        if self.save_checkpoints:
            self.save_checkpoint('final_model.pt')

        total_time = time.time() - training_start
        print(f"\nTraining complete!")
        print(f"Total wall time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")


def main():
    # Load .env from project root so DISCORD_HOOK is available if set
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        try:
            for line in _env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip().strip("'\"")
                    if key == "DISCORD_HOOK" and value:
                        os.environ["DISCORD_HOOK"] = value
                        break
        except Exception as _e:
            print(f"[Discord] Could not read .env: {_e}", file=sys.stderr)
    if os.environ.get("DISCORD_HOOK"):
        print("[Discord] Webhook configured — notifications enabled", file=sys.stderr)
    else:
        print("[Discord] No webhook (set DISCORD_HOOK in .env to enable)", file=sys.stderr)

    parser = argparse.ArgumentParser(description='Train V6 Phase-First LM')
    parser.add_argument('--size', type=str, default='small-matched',
                        choices=['tiny', 'small', 'small-matched', 'medium', 'large', 'xl'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dataset', type=str, default='tinystories',
                        choices=['tinystories', 'wikitext103', 'pg19', 'mixed'],
                        help='Dataset for autoregressive training')
    parser.add_argument('--mix_spec', type=str, default='wikipedia:40,pg19:30,openwebtext:30',
                        help='Mixed corpus spec: "source:weight,..." (only used with --dataset mixed)')
    parser.add_argument('--max_tokens_per_source', type=int, default=None,
                        help='Max samples per source in mixed corpus (None=full)')
    parser.add_argument('--max_samples', type=int, default=20000)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--no_working_memory', action='store_true',
                        help='Disable working memory (ablation)')
    parser.add_argument('--no_internal_memory', action='store_true',
                        help='Disable internal memory (ablation)')
    parser.add_argument('--wm_slots', type=int, default=None,
                        help='Override number of working memory slots')
    parser.add_argument('--im_slots', type=int, default=None,
                        help='Override number of internal memory slots')
    parser.add_argument('--init_strategy', type=str, default=None,
                        choices=list_strategies())
    parser.add_argument('--init_seed', type=int, default=None)
    parser.add_argument('--use_attention', action='store_true',
                        help='Enable PhaseAttention layers (disabled by default)')
    parser.add_argument('--attn_every', type=int, default=0,
                        help='Place attention every N layers (0 = last layer only)')
    parser.add_argument('--gen_every', type=int, default=0,
                        help='Generate a sample every N batches during training (0 = end of epoch only)')
    parser.add_argument('--gen_prompt', type=str, default='The',
                        help='Prompt for mid-epoch and end-of-epoch text generation')
    parser.add_argument('--lr_schedule', type=str, default=None,
                        choices=['cosine', 'warmup_cosine'],
                        help='LR schedule (default: from config)')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Warmup steps for warmup_cosine schedule')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Override dropout rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Override weight decay')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Max validation samples (default: max_samples // 10)')
    parser.add_argument('--no_text_repair', action='store_true',
                        help='Skip mojibake text repair on TinyStories')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable token cache (re-tokenize every run)')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--fullgraph', action='store_true',
                        help='Use fullgraph=True for torch.compile')
    parser.add_argument('--amp_dtype', type=str, default='auto',
                        choices=['auto', 'bf16', 'fp16'],
                        help='AMP dtype (auto prefers bf16 on capable hardware)')
    parser.add_argument('--no_tf32', action='store_true',
                        help='Disable TF32 matmul')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader worker count (default: from config)')
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Disable pinned memory for DataLoader')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Batch logging interval')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v6')
    parser.add_argument('--resume', type=str, default=None)

    # Training objective
    parser.add_argument('--objective', type=str, default='next_token',
                        choices=['next_token', 'span_corruption', 'delayed_recall'],
                        help='Training objective: next_token (standard AR), '
                             'span_corruption (T5-style infilling), '
                             'delayed_recall (fact-then-cue)')
    parser.add_argument('--span_corruption_rate', type=float, default=0.15)
    parser.add_argument('--span_mean_length', type=int, default=3)
    parser.add_argument('--delayed_recall_gap', type=int, default=64)

    # Episodic memory
    parser.add_argument('--episodic_slots', type=int, default=None,
                        help='Episodic memory slots (0 = disabled)')

    # Bank role training
    parser.add_argument('--bank_role_weight', type=float, default=None,
                        help='Weight for bank role specialization loss')

    # Diffusion / mode arguments
    parser.add_argument('--mode', type=str, default='autoregressive',
                        choices=['autoregressive', 'diffusion_text', 'diffusion_image', 'two_pass'])
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default='cosine',
                        choices=['cosine', 'linear'])
    parser.add_argument('--prediction_target', type=str, default='x0',
                        choices=['x0', 'epsilon'])
    parser.add_argument('--sampling_method', type=str, default='ddpm',
                        choices=['ddpm', 'ddim'])
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_encoder', type=str, default='patch',
                        choices=['patch', 'fft'])
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--image_dataset', type=str, default='tiny_imagenet')

    args = parser.parse_args()

    config = get_config(args.size)
    config.max_epochs = args.epochs
    config.mode = args.mode
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.no_working_memory:
        config.num_wm_slots = 0
    if args.no_internal_memory:
        config.num_im_slots = 0
    if args.wm_slots is not None:
        config.num_wm_slots = args.wm_slots
    if args.im_slots is not None:
        config.num_im_slots = args.im_slots
    if args.init_strategy is not None:
        config.init_strategy = args.init_strategy
    config.init_seed = args.init_seed
    if args.lr_schedule is not None:
        config.lr_schedule = args.lr_schedule
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.use_attention:
        config.use_attention = True
        config.attn_every = args.attn_every
    if args.compile:
        config.compile_model = True
    config.compile_mode = args.compile_mode
    config.compile_fullgraph = args.fullgraph
    config.amp_dtype = args.amp_dtype
    if args.no_tf32:
        config.allow_tf32 = False
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.no_pin_memory:
        config.pin_memory = False

    # Objective config
    config.objective = args.objective
    config.span_corruption_rate = args.span_corruption_rate
    config.span_mean_length = args.span_mean_length
    config.delayed_recall_gap = args.delayed_recall_gap
    if args.episodic_slots is not None:
        config.num_episodic_slots = args.episodic_slots
    if args.bank_role_weight is not None:
        config.bank_role_weight = args.bank_role_weight

    # Diffusion config
    config.diffusion_steps = args.diffusion_steps
    config.noise_schedule = args.noise_schedule
    config.prediction_target = args.prediction_target
    config.sampling_method = args.sampling_method
    config.image_size = args.image_size
    config.image_encoder = args.image_encoder
    config.patch_size = args.patch_size
    config.image_dataset = args.image_dataset

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = args.mode.replace('_', '-')
    log_path = log_dir / f'v6_{mode_tag}_{args.size}.log'
    log_mode = 'a' if args.resume else 'w'
    tee = TeeLogger(log_path, mode=log_mode)
    sys.stdout = tee

    if args.resume:
        print(f"\n--- Resumed from {args.resume} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    print(f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"V6 Phase-First Model (mode: {config.mode})")
    print("=" * 60)
    print(f"Size: {args.size}")
    if config.mode in ('autoregressive', 'diffusion_text', 'two_pass'):
        print(f"Dataset: {args.dataset}")
    print(f"Complex dim: {config.dim} (= {config.dim * 2} real values/position)")
    print(f"SSM state dim: {config.state_dim} (multi-timescale: fast/medium/slow)")
    print(f"Layers: {config.num_layers}")
    print(f"Banks: {config.num_banks} (semantic + context)")
    if config.mode == 'autoregressive':
        print(f"Working memory slots: {config.num_wm_slots} (top-k={config.wm_read_topk}, decay={config.wm_slot_decay})")
        print(f"Internal memory slots: {config.num_im_slots} (top-k={config.im_read_topk})")
    if config.mode.startswith('diffusion'):
        print(f"Diffusion steps: {config.diffusion_steps}, schedule: {config.noise_schedule}")
        print(f"Prediction target: {config.prediction_target}")
    if config.mode == 'diffusion_image':
        print(f"Image: {config.image_size}x{config.image_size}, encoder={config.image_encoder}, patch={config.patch_size}")
        print(f"Dataset: {config.image_dataset}")
    if config.use_attention:
        attn_desc = f"every {config.attn_every} layers" if config.attn_every > 0 else "last layer only"
        print(f"PhaseAttention: ENABLED ({attn_desc}, heads={config.attn_num_heads}, window={config.attn_window_size})")
    else:
        print(f"PhaseAttention: DISABLED (attention-free)")
    print(f"Epochs: {config.max_epochs}")
    print(f"LR schedule: {config.lr_schedule} (warmup={config.warmup_steps})")
    print(f"Dropout: {config.dropout}, Weight decay: {config.weight_decay}")

    amp_dtype = _resolve_amp_dtype(config.amp_dtype)
    amp_label = {torch.bfloat16: 'bf16', torch.float16: 'fp16'}.get(amp_dtype, 'off')
    print(f"AMP: {amp_label}, TF32: {config.allow_tf32}, Compile: {config.compile_model}"
          + (f" (mode={config.compile_mode})" if config.compile_model else ""))
    print(f"Workers: {config.num_workers}, Pin memory: {config.pin_memory}")
    print(f"Batch log interval: {args.log_interval}")
    print(f"Log file: {log_path}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 60)

    # CUDA performance settings
    if torch.cuda.is_available() and config.allow_tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Data loading
    tokenizer = None
    if config.mode in ('autoregressive', 'diffusion_text', 'two_pass'):
        if args.dataset == 'tinystories':
            train_ds, val_ds, tokenizer = load_tinystories(
                max_samples=args.max_samples, seq_len=args.seq_len,
                text_repair=not args.no_text_repair,
                use_cache=not args.no_cache,
                max_val_samples=args.max_val_samples,
            )
        elif args.dataset == 'wikitext103':
            train_ds, val_ds, tokenizer = load_wikitext103(
                max_samples=args.max_samples if args.max_samples < 9999999 else None,
                seq_len=args.seq_len,
                use_cache=not args.no_cache,
                max_val_samples=args.max_val_samples,
            )
        elif args.dataset == 'pg19':
            train_ds, val_ds, tokenizer = load_pg19(
                max_samples=args.max_samples if args.max_samples < 9999999 else None,
                seq_len=args.seq_len,
                use_cache=not args.no_cache,
                max_val_samples=args.max_val_samples,
            )
        elif args.dataset == 'mixed':
            train_ds, val_ds, tokenizer = load_mixed(
                mix_spec=args.mix_spec,
                seq_len=args.seq_len,
                max_tokens_per_source=args.max_tokens_per_source,
                use_cache=not args.no_cache,
            )
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        config.vocab_size = tokenizer.vocab_size
        config.max_seq_len = args.seq_len
    elif config.mode == 'diffusion_image':
        train_ds, val_ds = load_image_dataset(config)

    if config.mode in ('autoregressive', 'diffusion_text', 'two_pass') and config.objective != 'next_token':
        train_tokens = train_ds.data.reshape(-1)
        val_tokens = val_ds.data.reshape(-1) if hasattr(val_ds, 'data') else None
        if config.objective == 'span_corruption':
            print(f"Objective: span corruption (rate={config.span_corruption_rate}, "
                  f"mean_len={config.span_mean_length})")
            train_ds = SpanCorruptionDataset(
                train_tokens, seq_len=args.seq_len,
                corruption_rate=config.span_corruption_rate,
                mean_length=config.span_mean_length,
            )
            if val_tokens is not None:
                val_ds = SpanCorruptionDataset(
                    val_tokens, seq_len=args.seq_len,
                    corruption_rate=config.span_corruption_rate,
                    mean_length=config.span_mean_length,
                    seed=137,
                )
        elif config.objective == 'delayed_recall':
            print(f"Objective: delayed recall (gap={config.delayed_recall_gap})")
            train_ds = DelayedRecallDataset(
                train_tokens, seq_len=args.seq_len,
                gap=config.delayed_recall_gap,
            )
            if val_tokens is not None:
                val_ds = DelayedRecallDataset(
                    val_tokens, seq_len=args.seq_len,
                    gap=config.delayed_recall_gap,
                    seed=137,
                )

    use_cuda = torch.cuda.is_available()
    nw = config.num_workers if use_cuda else 0
    pm = config.pin_memory and use_cuda
    dl_kwargs = {}
    if nw > 0:
        dl_kwargs['persistent_workers'] = True
        dl_kwargs['prefetch_factor'] = 4
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True,
        num_workers=nw, pin_memory=pm,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False,
        num_workers=nw, pin_memory=pm,
        **dl_kwargs,
    )

    model = create_model(config)
    init_info = model.initializer_info
    print(f"Init strategy: {init_info['init_strategy']} (seed: {init_info['init_seed']})")

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    if config.compile_model:
        print(f"Compiling model with torch.compile "
              f"(mode={config.compile_mode}, fullgraph={config.compile_fullgraph})...")
        try:
            model = torch.compile(
                model, mode=config.compile_mode,
                fullgraph=config.compile_fullgraph,
            )
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without compilation")

    # Dispatch to the right trainer
    if config.mode in ('autoregressive', 'two_pass'):
        trainer = Trainer(
            model, config, train_loader, val_loader,
            tokenizer=tokenizer,
            checkpoint_dir=args.checkpoint_dir,
            start_epoch=start_epoch,
        )
        trainer.gen_every = args.gen_every
        trainer.gen_prompt = args.gen_prompt
        trainer.log_interval = args.log_interval
        if args.resume and 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.global_step = checkpoint.get('global_step', 0)
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            trainer.best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))
    else:
        trainer = DiffusionTrainer(
            model, config, train_loader, val_loader,
            tokenizer=tokenizer,
            checkpoint_dir=args.checkpoint_dir,
            start_epoch=start_epoch,
        )
        trainer.gen_every = args.gen_every
        trainer.log_interval = args.log_interval
        if config.mode == 'diffusion_image':
            sample_dir = Path(args.log_dir) / 'v6' / 'samples'
            sample_dir.mkdir(parents=True, exist_ok=True)
            trainer.sample_dir = sample_dir
        if args.resume and 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.global_step = checkpoint.get('global_step', 0)
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    trainer.train()

    print(f"\nWall clock end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = tee._stdout
    tee.close()


if __name__ == '__main__':
    main()
