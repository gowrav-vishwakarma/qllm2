"""
V7 data loading and training utilities.

Self-contained copy from v6/train.py so v7 is fully isolated.
"""

import json
import math
import os
import re
import sys
import unicodedata
import zlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset


# ── Logging ──────────────────────────────────────────────────────────────────

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


# ── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int = 512):
        self.seq_len = seq_len
        n_chunks = len(tokens) // (seq_len + 1)
        self.data = tokens[: n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {'input_ids': chunk[:-1], 'labels': chunk[1:]}


class MaskedTextDataset(Dataset):
    """Fixed-length samples with optional per-token loss mask (SFT)."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        assert input_ids.shape == labels.shape == loss_mask.shape
        self.input_ids = input_ids
        self.labels = labels
        self.loss_mask = loss_mask

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
            'loss_mask': self.loss_mask[idx],
        }


class StreamingTokenChunkDataset(IterableDataset):
    """Stream tokenized text and yield non-overlapping (seq_len+1) chunks."""

    def __init__(
        self,
        text_iterator: Iterator[str],
        tokenizer,
        seq_len: int,
        *,
        max_tokens: Optional[int] = None,
        shuffle_buffer: int = 10_000,
    ):
        self.text_iterator = text_iterator
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        buffer: List[int] = []
        yielded_tokens = 0
        pending: List[dict] = []

        for text in self.text_iterator:
            if not text or not str(text).strip():
                continue
            ids = self.tokenizer.encode(str(text), add_special_tokens=False)
            if not ids:
                continue
            ids.append(self.tokenizer.eos_token_id)
            buffer.extend(ids)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                sample = {
                    'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                    'labels': torch.tensor(chunk[1:], dtype=torch.long),
                }
                if self.shuffle_buffer > 0:
                    pending.append(sample)
                    if len(pending) >= self.shuffle_buffer:
                        idx = torch.randint(len(pending), (1,)).item()
                        out = pending.pop(idx)
                        yielded_tokens += self.seq_len
                        yield out
                        if self.max_tokens and yielded_tokens >= self.max_tokens:
                            return
                else:
                    yielded_tokens += self.seq_len
                    yield sample
                    if self.max_tokens and yielded_tokens >= self.max_tokens:
                        return

        while pending:
            out = pending.pop()
            yielded_tokens += self.seq_len
            yield out
            if self.max_tokens and yielded_tokens >= self.max_tokens:
                return


def _random_dataset(vocab_size: int, seq_len: int, num_samples: int):
    tokens = torch.randint(1, vocab_size, (num_samples * (seq_len + 1),))
    return TextDataset(tokens, seq_len)


# ── Text Repair ──────────────────────────────────────────────────────────────

_MOJIBAKE_REPLACEMENTS = [
    ('\u00e2\u0080\u009c', '\u201c'),
    ('\u00e2\u0080\u009d', '\u201d'),
    ('\u00e2\u0080\u0098', '\u2018'),
    ('\u00e2\u0080\u0099', '\u2019'),
    ('\u00e2\u0080\u0093', '\u2013'),
    ('\u00e2\u0080\u0094', '\u2014'),
    ('\u00e2\u0080\u00a6', '\u2026'),
    ('\u00c2\u00a1', '\u00a1'),
    ('\u00c2\u00bf', '\u00bf'),
]

_MOJIBAKE_RE = re.compile(
    r'[\u00c2-\u00c3][\u0080-\u00bf]|'
    r'\u00e2[\u0080-\u00bf][\u0080-\u00bf]'
)


def repair_text(text: str) -> str:
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
    all_tokens = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)['input_ids']
        for ids in encoded:
            all_tokens.extend(ids)
            all_tokens.append(tokenizer.eos_token_id)
    return all_tokens


# ── Data Loaders ─────────────────────────────────────────────────────────────

_CACHE_VERSION = 2


def load_wikitext103(
    max_samples=None, seq_len=512, use_cache=True, max_val_samples=None,
):
    """Load WikiText-103. Same pipeline as v6 for apples-to-apples comparison."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    limit_tag = f"ms{max_samples}" if max_samples else "full"
    cache_tag = f"v{_CACHE_VERSION}_{limit_tag}_sl{seq_len}"

    def _process_split(split_name, limit):
        cache_path = (
            Path(".cache") / "v7_tokens" / f"wikitext103_{split_name}_{cache_tag}.pt"
        )
        if use_cache and cache_path.exists():
            cached = torch.load(cache_path, weights_only=False)
            tokens = cached['tokens']
            print(
                f"[cache] Loaded WikiText-103 {split_name} from {cache_path} "
                f"({len(tokens):,} tokens)"
            )
            return tokens

        from datasets import load_dataset

        print(f"Loading WikiText-103 {split_name} (limit={limit})...")
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split_name)
        lines = [item['text'] for item in ds]
        if limit:
            lines = lines[:limit]

        all_tokens = []
        chunk_size = 50000
        for start in range(0, len(lines), chunk_size):
            chunk_lines = lines[start : start + chunk_size]
            chunk_text = '\n'.join(chunk_lines)
            ids = tokenizer.encode(chunk_text, add_special_tokens=False)
            all_tokens.extend(ids)

        tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"  {split_name}: {len(lines)} lines, {len(tokens):,} tokens")

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {'tokens': tokens, 'cache_version': _CACHE_VERSION}, cache_path,
            )
            print(f"[cache] Saved {split_name} to {cache_path}")

        return tokens

    try:
        train_tokens = _process_split('train', max_samples)
        val_limit = max_val_samples or (
            max(max_samples // 10, 1000) if max_samples else None
        )
        val_tokens = _process_split('validation', val_limit)
    except Exception as e:
        print(f"Failed to load WikiText-103: {e}")
        print("Using random data as fallback.")
        return (
            _random_dataset(50257, seq_len, 1000),
            _random_dataset(50257, seq_len, 100),
            tokenizer,
        )

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)
    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
    return train_ds, val_ds, tokenizer


def load_wikitext103_val(
    seq_len: int = 512,
    use_cache: bool = True,
    max_val_samples: Optional[int] = None,
):
    """WikiText-103 validation only (eval anchor during web pretrain)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    cache_tag = f"v{_CACHE_VERSION}_full_sl{seq_len}"
    cache_path = Path('.cache') / 'v7_tokens' / f'wikitext103_validation_{cache_tag}.pt'
    if use_cache and cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        val_tokens = cached['tokens']
        print(
            f"[cache] Loaded WikiText-103 validation from {cache_path} "
            f"({len(val_tokens):,} tokens)"
        )
    else:
        from datasets import load_dataset

        print('Loading WikiText-103 validation...')
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
        lines = [item['text'] for item in ds]
        if max_val_samples:
            lines = lines[:max_val_samples]
        chunk_text = '\n'.join(lines)
        val_tokens = torch.tensor(
            tokenizer.encode(chunk_text, add_special_tokens=False), dtype=torch.long,
        )
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'tokens': val_tokens, 'cache_version': _CACHE_VERSION}, cache_path)

    val_ds = TextDataset(val_tokens, seq_len)
    print(f"Val chunks: {len(val_ds)}")
    return val_ds, tokenizer


def load_tinystories(
    max_samples=20000, seq_len=512, text_repair=True, use_cache=True,
    max_val_samples=None,
):
    """Load TinyStories for quick dev iterations."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    repair_tag = "r1" if text_repair else "r0"
    cache_tag = f"v{_CACHE_VERSION}_ms{max_samples}_sl{seq_len}_{repair_tag}"

    def _process_split(split_name, limit):
        cache_path = (
            Path(".cache") / "v7_tokens" / f"tinystories_{split_name}_{cache_tag}.pt"
        )
        if use_cache and cache_path.exists():
            cached = torch.load(cache_path, weights_only=False)
            tokens = cached['tokens']
            print(
                f"[cache] Loaded TinyStories {split_name} from {cache_path} "
                f"({len(tokens):,} tokens)"
            )
            return tokens

        from datasets import load_dataset

        print(f"Loading TinyStories {split_name} (limit={limit})...")
        ds = load_dataset('roneneldan/TinyStories', split=split_name)
        texts = [item['text'] for item in ds if item['text'].strip()]
        if limit:
            texts = texts[:limit]

        if text_repair:
            texts = [repair_text(t) for t in texts]

        print(f"  Tokenizing {len(texts)} {split_name} texts...")
        token_list = _tokenize_batch(texts, tokenizer)
        tokens = torch.tensor(token_list, dtype=torch.long)
        print(f"  {split_name}: {len(texts)} stories, {len(tokens):,} tokens")

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {'tokens': tokens, 'cache_version': _CACHE_VERSION}, cache_path,
            )
            print(f"[cache] Saved {split_name} to {cache_path}")

        return tokens

    try:
        train_tokens = _process_split('train', max_samples)
        val_limit = max_val_samples or max(max_samples // 10, 1000)
        val_tokens = _process_split('validation', val_limit)
    except Exception as e:
        print(f"Failed to load TinyStories: {e}")
        print("Using random data as fallback.")
        return (
            _random_dataset(50257, seq_len, 1000),
            _random_dataset(50257, seq_len, 100),
            tokenizer,
        )

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)
    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
    return train_ds, val_ds, tokenizer


# ── Chat / SFT formatting (ChatML over GPT-2 vocab) ──────────────────────────

IM_START = '<|im_start|>'
IM_END = '<|im_end|>'
DEFAULT_SYSTEM = 'You are a helpful assistant.'

# Bump when the chat template / masking / special tokens change (invalidates cache).
_CHAT_CACHE_VERSION = 3

_SFT_FUNCTION_MARKERS = (
    'function_call', 'tool_calls', 'tool_call', 'available tools',
    '<tool_call>', '"type": "function"', 'json schema',
)

_CHAT_TOKENIZER = None


def get_chat_tokenizer():
    """GPT-2 tokenizer extended with ChatML role markers (vocab 50259).

    Adds two special tokens (`<|im_start|>`, `<|im_end|>`). `<|im_end|>` is the
    learned end-of-turn, *distinct* from pad/eos `<|endoftext|>` (50256), so it
    survives the assistant-only loss mask and the model can learn to stop.
    """
    global _CHAT_TOKENIZER
    if _CHAT_TOKENIZER is not None:
        return _CHAT_TOKENIZER
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.add_special_tokens({'additional_special_tokens': [IM_START, IM_END]})
    tok.pad_token = tok.eos_token  # <|endoftext|> (50256), distinct from <|im_end|>
    _CHAT_TOKENIZER = tok
    return tok


def format_chat_prompt(user_text: str, system: str = DEFAULT_SYSTEM) -> str:
    """ChatML prompt string for inference (ends ready for the assistant turn)."""
    return (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{user_text.strip()}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def format_chat_messages(
    messages: List[dict],
    *,
    default_system: str = DEFAULT_SYSTEM,
) -> str:
    """Render a full conversation to ChatML training text (debug/inspection)."""
    sys_msg = default_system
    for msg in messages:
        if (msg.get('role') or '').strip().lower() == 'system' and (msg.get('content') or '').strip():
            sys_msg = (msg.get('content') or '').strip()
            break
    parts = [f"{IM_START}system\n{sys_msg}{IM_END}\n"]
    for msg in messages:
        role = (msg.get('role') or '').strip().lower()
        content = (msg.get('content') or '').strip()
        if role in ('system', '') or not content:
            continue
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    return ''.join(parts)


def _should_filter_smoltalk(
    messages: List[dict],
    source: str,
    sft_filter: str,
    *,
    max_chars: int = 12_000,
) -> bool:
    if sft_filter == 'none':
        return False
    src = (source or '').lower()
    if 'function' in src or 'tool' in src:
        return True
    if 'magpie-ultra' in src and 'short' not in src:
        return True
    total_chars = sum(len(m.get('content') or '') for m in messages)
    if total_chars > max_chars:
        return True
    blob = json.dumps(messages, ensure_ascii=False).lower()
    if any(marker in blob for marker in _SFT_FUNCTION_MARKERS):
        return True
    if len(messages) > 8:
        return True
    return False


def _encode_sft_example(
    messages: List[dict],
    tokenizer,
    seq_len: int,
    *,
    default_system: str = DEFAULT_SYSTEM,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """ChatML-encode one conversation with assistant-only loss.

    Token ids are built turn-by-turn (robust to special tokens). Every assistant
    *content* token and its closing `<|im_end|>` is a training target; system,
    user, role-headers, and padding are masked out. This fixes two prior bugs:
      * the end-of-turn token used to equal pad and was stripped from the loss;
      * a single contiguous mask leaked later user turns into the loss.
    """
    im_start = tokenizer.convert_tokens_to_ids(IM_START)
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    nl = tokenizer.encode('\n', add_special_tokens=False)

    def enc(text: str) -> List[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    sys_msg = default_system
    for msg in messages:
        if (msg.get('role') or '').strip().lower() == 'system' and (msg.get('content') or '').strip():
            sys_msg = (msg.get('content') or '').strip()
            break

    ids: List[int] = []
    tgt: List[int] = []

    def add(tok_ids: List[int], is_target: bool):
        ids.extend(tok_ids)
        tgt.extend([1 if is_target else 0] * len(tok_ids))

    # System turn (context only).
    add([im_start], False)
    add(enc('system\n' + sys_msg), False)
    add([im_end], False)
    add(nl, False)

    has_assistant = False
    for msg in messages:
        role = (msg.get('role') or '').strip().lower()
        content = (msg.get('content') or '').strip()
        if role in ('system', '') or not content:
            continue
        is_assistant = role == 'assistant'
        add([im_start], False)
        add(enc(f"{role}\n"), False)        # role header is context, not a target
        add(enc(content), is_assistant)      # assistant content is the loss target
        add([im_end], is_assistant)          # learned end-of-turn
        add(nl, False)
        has_assistant = has_assistant or is_assistant

    if not has_assistant or len(ids) < 8:
        return None

    pad_id = tokenizer.pad_token_id
    if len(ids) > seq_len + 1:
        ids = ids[: seq_len + 1]
        tgt = tgt[: seq_len + 1]
    else:
        pad_n = (seq_len + 1) - len(ids)
        ids = ids + [pad_id] * pad_n
        tgt = tgt + [0] * pad_n

    input_ids = torch.tensor(ids[:-1], dtype=torch.long)
    labels = torch.tensor(ids[1:], dtype=torch.long)
    # loss_mask aligns to predicted (label) tokens: train where the *target* is assistant.
    loss_mask = torch.tensor(tgt[1:], dtype=torch.long)
    if loss_mask.sum() == 0:
        return None
    return input_ids, labels, loss_mask


def _conv_holdout_bucket(messages: List[dict], pct: int = 2) -> bool:
    """Deterministic ~pct% holdout by conversation-content hash."""
    blob = json.dumps(messages, ensure_ascii=False)[:4096]
    return (zlib.adler32(blob.encode('utf-8', errors='ignore')) % 100) >= (100 - pct)


def load_chat_sft(
    dataset_name: str,
    row_iter_fn,
    *,
    seq_len: int = 2048,
    sft_filter: str = 'hard',
    holdout_pct: int = 2,
    max_samples: Optional[int] = None,
    use_cache: bool = True,
):
    """Generic chat-SFT loader: ChatML encode + assistant-only mask + in-distro holdout.

    `row_iter_fn` is a callable returning an iterable of dicts with `messages`
    (and optional `source`). Returns `(train_ds, val_ds, tokenizer)` where val is
    a deterministic in-distribution holdout (not WikiText).
    """
    tokenizer = get_chat_tokenizer()
    filter_tag = sft_filter or 'none'
    limit_tag = f"ms{max_samples}" if max_samples else "full"
    cache_path = (
        Path('.cache') / 'v7_tokens'
        / f"{dataset_name}_{filter_tag}_{limit_tag}_sl{seq_len}_chatv{_CHAT_CACHE_VERSION}.pt"
    )

    if use_cache and cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        print(
            f"[cache] Loaded {dataset_name} chat SFT from {cache_path} "
            f"(train {cached['n_train']:,}, val {cached['n_val']:,})"
        )
        train_ds = MaskedTextDataset(cached['tr_input'], cached['tr_labels'], cached['tr_mask'])
        val_ds = MaskedTextDataset(cached['va_input'], cached['va_labels'], cached['va_mask'])
        return train_ds, val_ds, tokenizer

    print(f"Building {dataset_name} chat SFT (filter={filter_tag}, limit={max_samples})...")
    tr: Tuple[list, list, list] = ([], [], [])
    va: Tuple[list, list, list] = ([], [], [])
    kept = skipped = held = 0
    for ex in row_iter_fn():
        messages = ex.get('messages') or []
        source = ex.get('source') or ''
        if _should_filter_smoltalk(messages, source, filter_tag, max_chars=seq_len * 6):
            skipped += 1
            continue
        enc = _encode_sft_example(messages, tokenizer, seq_len)
        if enc is None:
            skipped += 1
            continue
        bucket = va if _conv_holdout_bucket(messages, holdout_pct) else tr
        bucket[0].append(enc[0])
        bucket[1].append(enc[1])
        bucket[2].append(enc[2])
        kept += 1
        if bucket is va:
            held += 1
        if kept % 5000 == 0:
            print(
                f"  {dataset_name}: kept {kept:,} (val {held:,}), skipped {skipped:,}...",
                flush=True,
            )

    if not tr[0]:
        raise RuntimeError(f"{dataset_name}: filtering removed all training samples")
    if not va[0]:
        for _ in range(min(64, len(tr[0]) - 1)):
            for k in range(3):
                va[k].append(tr[k].pop())

    def _stack(triple):
        return (
            torch.stack(triple[0]),
            torch.stack(triple[1]),
            torch.stack(triple[2]),
        )

    tr_input, tr_labels, tr_mask = _stack(tr)
    va_input, va_labels, va_mask = _stack(va)
    train_ds = MaskedTextDataset(tr_input, tr_labels, tr_mask)
    val_ds = MaskedTextDataset(va_input, va_labels, va_mask)
    print(
        f"  {dataset_name}: train {len(train_ds):,}, val {len(val_ds):,}, "
        f"skipped {skipped:,}, shape={tuple(tr_input.shape)}"
    )

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'tr_input': tr_input, 'tr_labels': tr_labels, 'tr_mask': tr_mask,
                'va_input': va_input, 'va_labels': va_labels, 'va_mask': va_mask,
                'n_train': len(train_ds), 'n_val': len(val_ds),
                'chat_cache_version': _CHAT_CACHE_VERSION,
            },
            cache_path,
        )
        print(f"[cache] Saved {dataset_name} chat SFT to {cache_path}")

    return train_ds, val_ds, tokenizer


def _dclm_edu_text_iter(edu_score_min: int = 3) -> Iterator[str]:
    """Stream DCLM-Edu text filtered by integer edu score."""
    from datasets import load_dataset

    stream = load_dataset('HuggingFaceTB/dclm-edu', split='train', streaming=True)
    for row in stream:
        score = row.get('edu_int_score')
        if score is not None and score < edu_score_min:
            continue
        text = row.get('text') or row.get('content') or ''
        if text.strip():
            yield text


def _fineweb_edu_text_iter(edu_score_min: int = 3, name: str = 'sample-10BT') -> Iterator[str]:
    """Stream FineWeb-Edu text filtered by the edu classifier score."""
    from datasets import load_dataset

    stream = load_dataset(
        'HuggingFaceFW/fineweb-edu', name=name, split='train', streaming=True,
    )
    for row in stream:
        score = row.get('score')
        if score is None:
            score = row.get('edu_score')
        if score is not None and score < edu_score_min:
            continue
        text = row.get('text') or row.get('content') or ''
        if text.strip():
            yield text


def _interleave_text_iters(iters, weights=None) -> Iterator[str]:
    """Weighted round-robin over multiple text iterators until all exhaust."""
    import random as _random

    iters = [iter(it) for it in iters]
    weights = list(weights) if weights else [1.0] * len(iters)
    alive = [True] * len(iters)
    while any(alive):
        idxs = [i for i, a in enumerate(alive) if a]
        w = [weights[i] for i in idxs]
        choice = _random.choices(idxs, weights=w, k=1)[0]
        try:
            yield next(iters[choice])
        except StopIteration:
            alive[choice] = False


def load_dclm_edu(
    seq_len: int = 2048,
    edu_score_min: int = 3,
    token_budget: Optional[int] = None,
    max_val_samples: Optional[int] = None,
):
    """Stream DCLM-Edu for pretrain; WikiText-103 validation for eval anchor."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Streaming DCLM-Edu (edu_int_score>={edu_score_min}, "
        f"token_budget={token_budget or 'none'})..."
    )
    train_ds = StreamingTokenChunkDataset(
        _dclm_edu_text_iter(edu_score_min),
        tokenizer,
        seq_len,
        max_tokens=token_budget,
        shuffle_buffer=10_000,
    )

    val_ds, _ = load_wikitext103_val(
        seq_len=seq_len,
        max_val_samples=max_val_samples,
    )
    return train_ds, val_ds, tokenizer


def load_pretrain_mix(
    seq_len: int = 2048,
    edu_score_min: int = 3,
    token_budget: Optional[int] = None,
    sources: Tuple[str, ...] = ('dclm', 'fineweb'),
    weights: Optional[Tuple[float, ...]] = None,
    chat_vocab: bool = True,
    fineweb_name: str = 'sample-10BT',
    max_val_samples: Optional[int] = None,
):
    """Stream a mixed web-edu corpus (DCLM-Edu + FineWeb-Edu) for from-scratch pretrain.

    With ``chat_vocab=True`` the ChatML tokenizer (vocab 50259) is used so the
    chat special-token embeddings exist from initialization (no later resize).
    WikiText-103 val remains the tracking anchor.
    """
    if chat_vocab:
        tokenizer = get_chat_tokenizer()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

    iters = []
    for s in sources:
        if s == 'dclm':
            iters.append(_dclm_edu_text_iter(edu_score_min))
        elif s == 'fineweb':
            iters.append(_fineweb_edu_text_iter(edu_score_min, name=fineweb_name))
        else:
            raise ValueError(f"Unknown pretrain source: {s}")

    print(
        f"Streaming pretrain mix: sources={list(sources)} "
        f"weights={list(weights) if weights else 'uniform'} edu>={edu_score_min} "
        f"budget={token_budget or 'none'} vocab={len(tokenizer)}"
    )
    text_iter = _interleave_text_iters(iters, weights) if len(iters) > 1 else iters[0]
    train_ds = StreamingTokenChunkDataset(
        text_iter,
        tokenizer,
        seq_len,
        max_tokens=token_budget,
        shuffle_buffer=10_000,
    )
    val_ds, _ = load_wikitext103_val(seq_len=seq_len, max_val_samples=max_val_samples)
    return train_ds, val_ds, tokenizer


def load_fineweb_edu(
    seq_len: int = 2048,
    edu_score_min: int = 3,
    token_budget: Optional[int] = None,
    chat_vocab: bool = False,
    fineweb_name: str = 'sample-10BT',
    max_val_samples: Optional[int] = None,
):
    """Stream FineWeb-Edu only (mirror of load_dclm_edu)."""
    return load_pretrain_mix(
        seq_len=seq_len, edu_score_min=edu_score_min, token_budget=token_budget,
        sources=('fineweb',), chat_vocab=chat_vocab, fineweb_name=fineweb_name,
        max_val_samples=max_val_samples,
    )


def load_smoltalk2(
    seq_len: int = 2048,
    max_samples: Optional[int] = None,
    sft_filter: str = 'hard',
    use_cache: bool = True,
    **_unused,
):
    """Filtered SmolTalk chat SFT (ChatML, assistant-only loss, in-distro holdout val)."""
    def _rows():
        from datasets import load_dataset

        ds = load_dataset('HuggingFaceTB/smol-smoltalk', split='train')
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        for ex in ds:
            yield ex

    return load_chat_sft(
        'smoltalk2', _rows,
        seq_len=seq_len, sft_filter=sft_filter,
        max_samples=max_samples, use_cache=use_cache,
    )


# Heuristic source->category map for the Tulu-3 SFT mixture. Source strings vary;
# matching is substring-based and refined when the run actually executes.
TULU3_CATEGORY_SOURCES = {
    'coding': ('code', 'coding', 'codealpaca'),
    'knowledge_recall': ('flan', 'no_robots', 'oasst', 'open_assistant', 'sciriff', 'table'),
    'general': ('wildchat', 'persona', 'lima', 'hardcoded', 'tulu', 'if_', 'ifeval'),
}
TULU3_EXCLUDE = (
    'math', 'gsm', 'numina', 'aya', 'translat', 'multiling',
    'safety', 'jailbreak', 'wildguard',
)


def _tulu3_keep_source(source: str, categories, exclude) -> bool:
    s = (source or '').lower()
    if any(x in s for x in exclude):
        return False
    if not categories:
        return True
    wanted = []
    for c in categories:
        wanted += list(TULU3_CATEGORY_SOURCES.get(c, ()))
    if not wanted:
        return True
    return any(w in s for w in wanted)


def load_tulu3_sft(
    seq_len: int = 2048,
    max_samples: Optional[int] = None,
    sft_filter: str = 'hard',
    categories: Tuple[str, ...] = ('knowledge_recall', 'coding', 'general'),
    use_cache: bool = True,
    **_unused,
):
    """Tulu-3 SFT mixture (knowledge+coding+general) -> ChatML assistant-only SFT."""
    cats = tuple(categories) if categories else ()
    cat_tag = '+'.join(cats) if cats else 'all'

    def _rows():
        from datasets import load_dataset

        ds = load_dataset('allenai/tulu-3-sft-mixture', split='train')
        n = 0
        for ex in ds:
            if not _tulu3_keep_source(ex.get('source'), cats, TULU3_EXCLUDE):
                continue
            yield ex
            n += 1
            if max_samples and n >= max_samples:
                break

    return load_chat_sft(
        f'tulu3_{cat_tag}', _rows,
        seq_len=seq_len, sft_filter=sft_filter,
        max_samples=max_samples, use_cache=use_cache,
    )


# ── Evaluation Metrics ───────────────────────────────────────────────────────

def compute_text_quality(text: str) -> Dict[str, float]:
    """Behavioral quality metrics beyond perplexity."""
    words = text.split()
    n_words = max(len(words), 1)

    def ngram_repeat_rate(tokens, n):
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        return 1.0 - len(set(ngrams)) / max(len(ngrams), 1)

    restart_patterns = ['once upon a time', '<|endoftext|>']
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


# ── Training Utilities ───────────────────────────────────────────────────────

def resolve_amp_dtype(amp_dtype_str: str) -> Optional[torch.dtype]:
    if not torch.cuda.is_available():
        return None
    if amp_dtype_str == 'bf16':
        return torch.bfloat16
    if amp_dtype_str == 'fp16':
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_lr_scheduler(
    optimizer, schedule: str, warmup_steps: int, total_steps: int,
):
    total_steps = max(total_steps, 1)
    if schedule == 'warmup_cosine' and warmup_steps > 0:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


_NO_DECAY_SUFFIXES = {'dt_bias'}


def build_param_groups(model: nn.Module, weight_decay: float):
    """Split into decay/no-decay groups for AdamW.

    Decay: 2-D+ weight matrices (ComplexLinear weight_real/weight_imag, nn.Linear weight).
    No-decay: biases, norms, scalars, dt_bias, protect_gate bias.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        suffix = name.split('.')[-1]
        if suffix in _NO_DECAY_SUFFIXES:
            no_decay.append(param)
        elif suffix in ('bias', 'bias_real', 'bias_imag'):
            no_decay.append(param)
        elif param.dim() >= 2 and suffix in ('weight', 'weight_real', 'weight_imag'):
            decay.append(param)
        else:
            no_decay.append(param)

    n_d = sum(p.numel() for p in decay)
    n_nd = sum(p.numel() for p in no_decay)
    print(
        f"Param groups: {len(decay)} tensors ({n_d:,}) with wd, "
        f"{len(no_decay)} tensors ({n_nd:,}) without"
    )
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
