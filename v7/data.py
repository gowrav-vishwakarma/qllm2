"""
V7 data loading and training utilities.

Self-contained copy from v6/train.py so v7 is fully isolated.
"""

import json
import hashlib
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
from torch.utils.data import ConcatDataset, Dataset, IterableDataset


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


class StackedChunkDataset(Dataset):
    """Fixed (input_ids, labels) chunks for cached pretrain streams."""

    def __init__(self, input_ids: torch.Tensor, labels: torch.Tensor):
        assert input_ids.shape == labels.shape
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'labels': self.labels[idx]}


class StreamingTokenChunkDataset(IterableDataset):
    """Stream tokenized text and yield non-overlapping (seq_len+1) chunks.

    ``text_iterator`` may yield plain strings or ``(source_name, text)`` tuples.
    When ``token_counters`` is provided, increments per-source token counts.
    """

    def __init__(
        self,
        text_iterator: Iterator[str],
        tokenizer,
        seq_len: int,
        *,
        max_tokens: Optional[int] = None,
        shuffle_buffer: int = 10_000,
        token_counters: Optional[Dict[str, int]] = None,
        default_source: str = 'mix',
    ):
        self.text_iterator = text_iterator
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.shuffle_buffer = shuffle_buffer
        self.token_counters = token_counters
        self.default_source = default_source

    def _yield_chunk(self, yielded_tokens: int, source: str) -> int:
        if self.token_counters is not None:
            self.token_counters[source] = self.token_counters.get(source, 0) + self.seq_len
        return yielded_tokens + self.seq_len

    def __iter__(self):
        buffer: List[int] = []
        yielded_tokens = 0
        pending: List[dict] = []

        for item in self.text_iterator:
            if isinstance(item, tuple):
                source, text = item
            else:
                source, text = self.default_source, item
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
                    pending.append((source, sample))
                    if len(pending) >= self.shuffle_buffer:
                        idx = torch.randint(len(pending), (1,)).item()
                        src, out = pending.pop(idx)
                        yielded_tokens = self._yield_chunk(yielded_tokens, src)
                        yield out
                        if self.max_tokens and yielded_tokens >= self.max_tokens:
                            return
                else:
                    yielded_tokens = self._yield_chunk(yielded_tokens, source)
                    yield sample
                    if self.max_tokens and yielded_tokens >= self.max_tokens:
                        return

        while pending:
            src, out = pending.pop()
            yielded_tokens = self._yield_chunk(yielded_tokens, src)
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
THINK_START = '<think>'
THINK_END = '</think>'
DEFAULT_SYSTEM = 'You are a helpful assistant.'

# Bump when the chat template / masking / special tokens change (invalidates cache).
# v4: added <think>/</think> reasoning specials (vocab 50259 -> 50261).
_CHAT_CACHE_VERSION = 4

_SFT_FUNCTION_MARKERS = (
    'function_call', 'tool_calls', 'tool_call', 'available tools',
    '<tool_call>', '"type": "function"', 'json schema',
)

_CHAT_TOKENIZER = None


def get_chat_tokenizer():
    """GPT-2 tokenizer extended with ChatML + reasoning markers (vocab 50261).

    Adds four special tokens: `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`.
    `<|im_end|>` is the learned end-of-turn, *distinct* from pad/eos
    `<|endoftext|>` (50256), so it survives the assistant-only loss mask and the
    model can learn to stop. `<think>`/`</think>` are single, stable reasoning
    boundaries (instead of a fragile multi-token BPE sequence) so the model can
    reliably open/close a reasoning block; trained from step 0 via the pretrain
    reasoning blend.
    """
    global _CHAT_TOKENIZER
    if _CHAT_TOKENIZER is not None:
        return _CHAT_TOKENIZER
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.add_special_tokens(
        {'additional_special_tokens': [IM_START, IM_END, THINK_START, THINK_END]}
    )
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


def format_chat_prompt_from_messages(
    messages: List[dict],
    *,
    default_system: str = DEFAULT_SYSTEM,
) -> str:
    """ChatML prompt for inference on a multi-turn conversation."""
    return format_chat_messages(messages, default_system=default_system) + f"{IM_START}assistant\n"


_ROLE_ALIASES = {
    'human': 'user', 'gpt': 'assistant', 'bot': 'assistant', 'chatbot': 'assistant',
    'system': 'system', 'user': 'user', 'assistant': 'assistant', 'tool': 'tool',
}


def _normalize_messages(raw) -> List[dict]:
    """Normalize varied chat schemas to a list of {role, content} dicts.

    Handles OpenAI-style (`messages` with role/content) and ShareGPT-style
    (`conversations` with from/value). Unknown roles are dropped.
    """
    if not raw:
        return []
    out: List[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get('role') or m.get('from') or ''
        content = m.get('content')
        if content is None:
            content = m.get('value') or ''
        role = _ROLE_ALIASES.get(str(role).strip().lower(), str(role).strip().lower())
        if not content or not str(content).strip():
            continue
        out.append({'role': role, 'content': str(content)})
    return out


def _messages_have_think(messages: List[dict]) -> bool:
    """True if any assistant turn contains a <think> reasoning block."""
    for m in messages:
        if (m.get('role') or '').lower() == 'assistant' and THINK_START in (m.get('content') or ''):
            return True
    return False


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


# Flush train buffers to disk every N rows during SFT cache build so ~2M rows
# does not OOM (single-file stack peaked at ~165 GB on this host).
_SFT_CACHE_SHARD_SIZE = 50_000


def _chat_sft_cache_stem(dataset_name: str, filter_tag: str, limit_tag: str, seq_len: int) -> str:
    return f"{dataset_name}_{filter_tag}_{limit_tag}_sl{seq_len}_chatv{_CHAT_CACHE_VERSION}"


def _stack_sft_triple(triple: Tuple[list, list, list]):
    return (
        torch.stack(triple[0]),
        torch.stack(triple[1]),
        torch.stack(triple[2]),
    )


def _atomic_torch_save(obj, path: Path) -> None:
    """Write via temp file so a full disk mid-save cannot corrupt the target."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _read_sft_build_state(cache_dir: Path) -> Optional[dict]:
    """Resume metadata written after each flushed train shard."""
    state_path = cache_dir / 'build_state.json'
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return None


def _write_sft_build_state(cache_dir: Path, state: dict) -> None:
    path = cache_dir / 'build_state.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state) + '\n')
    os.replace(tmp, path)


def _validate_existing_sft_shards(cache_dir: Path) -> Tuple[int, int]:
    """Return (n_valid_shards, n_train_rows) dropping any corrupt tail shard."""
    n_shards = n_train = 0
    for path in sorted(cache_dir.glob('train_*.pt')):
        try:
            part = torch.load(path, map_location='cpu', weights_only=False)
            rows = int(part['input_ids'].shape[0])
        except Exception:
            path.unlink(missing_ok=True)
            print(f"  [cache] removed corrupt shard {path.name}", flush=True)
            break
        n_shards += 1
        n_train += rows
    return n_shards, n_train


def _load_chat_sft_from_sharded_cache(cache_dir: Path, dataset_name: str):
    manifest = json.loads((cache_dir / 'manifest.json').read_text())
    shards = []
    for i in range(manifest['n_train_shards']):
        part = torch.load(cache_dir / f'train_{i:04d}.pt', weights_only=False)
        shards.append(MaskedTextDataset(part['input_ids'], part['labels'], part['loss_mask']))
    val = torch.load(cache_dir / 'val.pt', weights_only=False)
    train_ds = shards[0] if len(shards) == 1 else ConcatDataset(shards)
    val_ds = MaskedTextDataset(val['input_ids'], val['labels'], val['loss_mask'])
    print(
        f"[cache] Loaded {dataset_name} chat SFT from {cache_dir} "
        f"(train {manifest['n_train']:,} in {manifest['n_train_shards']} shards, "
        f"val {manifest['n_val']:,})"
    )
    return train_ds, val_ds


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

    Large corpora are cached as on-disk shards (``train_0000.pt``, ...) so cache
    build stays bounded in RAM; legacy single ``.pt`` caches are still supported.
    """
    tokenizer = get_chat_tokenizer()
    filter_tag = sft_filter or 'none'
    limit_tag = f"ms{max_samples}" if max_samples else "full"
    cache_root = Path('.cache') / 'v7_tokens'
    cache_stem = _chat_sft_cache_stem(dataset_name, filter_tag, limit_tag, seq_len)
    cache_path = cache_root / f'{cache_stem}.pt'
    cache_dir = cache_root / cache_stem

    if use_cache and (cache_dir / 'manifest.json').exists():
        train_ds, val_ds = _load_chat_sft_from_sharded_cache(cache_dir, dataset_name)
        return train_ds, val_ds, tokenizer
    if use_cache and cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        print(
            f"[cache] Loaded {dataset_name} chat SFT from {cache_path} "
            f"(train {cached['n_train']:,}, val {cached['n_val']:,})"
        )
        train_ds = MaskedTextDataset(cached['tr_input'], cached['tr_labels'], cached['tr_mask'])
        val_ds = MaskedTextDataset(cached['va_input'], cached['va_labels'], cached['va_mask'])
        return train_ds, val_ds, tokenizer

    print(
        f"Building {dataset_name} chat SFT (filter={filter_tag}, limit={max_samples}, "
        f"shard={_SFT_CACHE_SHARD_SIZE:,})..."
    )
    tr: Tuple[list, list, list] = ([], [], [])
    va: Tuple[list, list, list] = ([], [], [])
    kept = skipped = held = n_train = 0
    train_shard = 0
    resume_kept = 0
    resume_train_rows = 0
    train_rows_seen = 0
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_shard, n_train = _validate_existing_sft_shards(cache_dir)
        state = _read_sft_build_state(cache_dir)
        if state and train_shard > 0:
            resume_kept = int(state.get('kept', 0))
            print(
                f"  [cache] resuming from kept={resume_kept:,}, "
                f"shards={train_shard}, train_rows={n_train:,}",
                flush=True,
            )
        elif train_shard > 0:
            resume_train_rows = n_train
            print(
                f"  [cache] resuming from {train_shard} shards "
                f"({n_train:,} train rows, no build_state)...",
                flush=True,
            )

    def _flush_train_shard():
        nonlocal train_shard, n_train, kept, held, skipped
        if not tr[0]:
            return
        inp, lab, mask = _stack_sft_triple(tr)
        n_train += inp.shape[0]
        if use_cache:
            shard_path = cache_dir / f'train_{train_shard:04d}.pt'
            _atomic_torch_save(
                {'input_ids': inp, 'labels': lab, 'loss_mask': mask},
                shard_path,
            )
            _write_sft_build_state(cache_dir, {
                'kept': kept,
                'held': held,
                'skipped': skipped,
                'train_shard': train_shard + 1,
                'n_train': n_train,
            })
        train_shard += 1
        tr[0].clear()
        tr[1].clear()
        tr[2].clear()

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
        in_holdout = _conv_holdout_bucket(messages, holdout_pct)
        if resume_kept > 0 and kept < resume_kept:
            if in_holdout:
                va[0].append(enc[0])
                va[1].append(enc[1])
                va[2].append(enc[2])
                held += 1
            kept += 1
            continue
        if resume_train_rows > 0 and train_rows_seen < resume_train_rows:
            if in_holdout:
                va[0].append(enc[0])
                va[1].append(enc[1])
                va[2].append(enc[2])
                held += 1
            else:
                train_rows_seen += 1
            kept += 1
            continue
        if in_holdout:
            va[0].append(enc[0])
            va[1].append(enc[1])
            va[2].append(enc[2])
            held += 1
        else:
            tr[0].append(enc[0])
            tr[1].append(enc[1])
            tr[2].append(enc[2])
            if len(tr[0]) >= _SFT_CACHE_SHARD_SIZE:
                _flush_train_shard()
        kept += 1
        if kept % 5000 == 0:
            print(
                f"  {dataset_name}: kept {kept:,} (train {n_train + len(tr[0]):,}, "
                f"val {held:,}), skipped {skipped:,}...",
                flush=True,
            )

    _flush_train_shard()
    if n_train == 0:
        raise RuntimeError(f"{dataset_name}: filtering removed all training samples")
    if not va[0]:
        raise RuntimeError(f"{dataset_name}: holdout val empty after filtering")

    va_input, va_labels, va_mask = _stack_sft_triple(va)
    print(
        f"  {dataset_name}: train {n_train:,}, val {len(va_input):,}, "
        f"skipped {skipped:,}, shards={train_shard}"
    )

    if use_cache:
        _atomic_torch_save(
            {'input_ids': va_input, 'labels': va_labels, 'loss_mask': va_mask},
            cache_dir / 'val.pt',
        )
        manifest = {
            'n_train': n_train,
            'n_val': int(va_input.shape[0]),
            'n_train_shards': train_shard,
            'seq_len': seq_len,
            'chat_cache_version': _CHAT_CACHE_VERSION,
        }
        (cache_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (cache_dir / 'build_state.json').unlink(missing_ok=True)
        print(f"[cache] Saved {dataset_name} chat SFT shards to {cache_dir}")
        train_ds, val_ds = _load_chat_sft_from_sharded_cache(cache_dir, dataset_name)
        return train_ds, val_ds, tokenizer

    val_ds = MaskedTextDataset(va_input, va_labels, va_mask)
    train_ds = MaskedTextDataset(*_stack_sft_triple(tr)) if tr[0] else None
    if train_ds is None:
        raise RuntimeError(f"{dataset_name}: no training samples after build")
    return train_ds, val_ds, tokenizer


_PRETRAIN_HOLDOUT_CACHE_VERSION = 1


def pretrain_holdout_bucket(text: str, pct: int = 5) -> bool:
    """Deterministic ~pct% holdout by document content hash."""
    sample = (text or '')[:8192]
    return (zlib.adler32(sample.encode('utf-8', errors='ignore')) % 100) >= (100 - pct)


def _bump_counter(counters: Optional[Dict[str, int]], key: str) -> None:
    if counters is not None and key:
        counters[key] = counters.get(key, 0) + 1


def _dclm_edu_text_iter(
    edu_score_min: int = 3,
    *,
    exclude_holdout: bool = False,
    holdout_pct: int = 5,
    only_holdout: bool = False,
    skip_docs: int = 0,
    doc_counters: Optional[Dict[str, int]] = None,
    counter_key: str = 'dclm',
) -> Iterator[str]:
    """Stream DCLM-Edu text filtered by integer edu score.

    When ``doc_counters`` is given, increments ``counter_key`` for each yielded
    (post-filter, post-skip) doc -- this is the monotonic cursor used to compute
    ``skip_docs`` for the next round so no doc is reused.
    """
    from datasets import load_dataset

    stream = load_dataset('HuggingFaceTB/dclm-edu', split='train', streaming=True)
    skipped = 0
    for row in stream:
        score = row.get('edu_int_score')
        if score is not None and score < edu_score_min:
            continue
        text = row.get('text') or row.get('content') or ''
        if not text.strip():
            continue
        in_holdout = pretrain_holdout_bucket(text, holdout_pct)
        if exclude_holdout and in_holdout:
            continue
        if only_holdout and not in_holdout:
            continue
        if skipped < skip_docs:
            skipped += 1
            continue
        _bump_counter(doc_counters, counter_key)
        yield text


def _fineweb_edu_text_iter(
    edu_score_min: int = 3,
    name: str = 'sample-10BT',
    *,
    exclude_holdout: bool = False,
    holdout_pct: int = 5,
    only_holdout: bool = False,
    skip_docs: int = 0,
    doc_counters: Optional[Dict[str, int]] = None,
    counter_key: str = 'fineweb',
) -> Iterator[str]:
    """Stream FineWeb-Edu text filtered by the edu classifier score."""
    from datasets import load_dataset

    stream = load_dataset(
        'HuggingFaceFW/fineweb-edu', name=name, split='train', streaming=True,
    )
    skipped = 0
    for row in stream:
        score = row.get('score')
        if score is None:
            score = row.get('edu_score')
        if score is not None and score < edu_score_min:
            continue
        text = row.get('text') or row.get('content') or ''
        if not text.strip():
            continue
        in_holdout = pretrain_holdout_bucket(text, holdout_pct)
        if exclude_holdout and in_holdout:
            continue
        if only_holdout and not in_holdout:
            continue
        if skipped < skip_docs:
            skipped += 1
            continue
        _bump_counter(doc_counters, counter_key)
        yield text


def _tag_source(name: str, it: Iterator[str]) -> Iterator[Tuple[str, str]]:
    for text in it:
        yield name, text


def _interleave_text_iters(
    tagged_iters: List[Tuple[str, Iterator[str]]],
    weights=None,
    *,
    seed: int = 42,
) -> Iterator[Tuple[str, str]]:
    """Weighted round-robin over tagged text iterators until all exhaust."""
    import random as _random

    rng = _random.Random(seed)
    names = [n for n, _ in tagged_iters]
    iters = [iter(it) for _, it in tagged_iters]
    weights = list(weights) if weights else [1.0] * len(iters)
    alive = [True] * len(iters)
    while any(alive):
        idxs = [i for i, a in enumerate(alive) if a]
        w = [weights[i] for i in idxs]
        choice = rng.choices(idxs, weights=w, k=1)[0]
        try:
            yield names[choice], next(iters[choice])
        except StopIteration:
            alive[choice] = False


def _blend_interleave_text_iters(
    tagged_iters: List[Tuple[str, Iterator[str]]],
    weights=None,
    *,
    warmup_tokens: int = 0,
    warmup_sources: Tuple[str, ...] = (),
    token_counters: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Iterator[Tuple[str, str]]:
    """Weighted round-robin with a grammar warmup.

    While total tokens consumed (``sum(token_counters.values())``) is below
    ``warmup_tokens``, only draw from ``warmup_sources`` (web/grammar). After the
    warmup, the full weighted blend applies. ``warmup_tokens=0`` disables warmup.
    """
    import random as _random

    rng = _random.Random(seed)
    names = [n for n, _ in tagged_iters]
    iters = [iter(it) for _, it in tagged_iters]
    weights = list(weights) if weights else [1.0] * len(iters)
    alive = [True] * len(iters)
    warmup_set = set(warmup_sources)
    can_warmup = warmup_tokens > 0 and token_counters is not None and bool(warmup_set)
    while any(alive):
        consumed = sum(token_counters.values()) if token_counters else 0
        in_warmup = can_warmup and consumed < warmup_tokens
        if in_warmup:
            cand = [i for i, a in enumerate(alive) if a and names[i] in warmup_set]
            if not cand:  # warmup sources exhausted early -> open the full blend
                cand = [i for i, a in enumerate(alive) if a]
        else:
            cand = [i for i, a in enumerate(alive) if a]
        w = [weights[i] for i in cand]
        choice = rng.choices(cand, weights=w, k=1)[0]
        try:
            yield names[choice], next(iters[choice])
        except StopIteration:
            alive[choice] = False


# ── smoltalk2 streaming (format-aware chat/reasoning) ────────────────────────

SMOLTALK2_REPO = 'HuggingFaceTB/smoltalk2'


def _smoltalk2_split_names(config: str) -> List[str]:
    """Discover split names for a smoltalk2 config (sorted for determinism)."""
    try:
        from datasets import get_dataset_split_names

        names = get_dataset_split_names(SMOLTALK2_REPO, config)
        if names:
            return sorted(names)
    except Exception as exc:  # noqa: BLE001 - fall back to a single 'train' split
        print(f"  [smoltalk2] split discovery failed for {config}: {exc}; trying 'train'")
    return ['train']


def _smoltalk2_row_iter(
    config: str = 'SFT',
    *,
    skip_rows: int = 0,
    doc_counters: Optional[Dict[str, int]] = None,
    counter_key: str = '',
    splits: Optional[List[str]] = None,
) -> Iterator[List[dict]]:
    """Stream normalized message lists from a smoltalk2 config.

    Splits are concatenated in sorted order (deterministic); ``skip_rows`` skips
    the first N eligible rows across the concatenation for cross-round freshness.
    """
    from datasets import load_dataset

    names = splits or _smoltalk2_split_names(config)
    skipped = 0
    for split in names:
        try:
            ds = load_dataset(SMOLTALK2_REPO, config, split=split, streaming=True)
        except Exception as exc:  # noqa: BLE001 - skip a split that fails to open
            print(f"  [smoltalk2] skip split {config}/{split}: {exc}")
            continue
        for row in ds:
            msgs = _normalize_messages(row.get('messages') or row.get('conversations'))
            if not msgs:
                continue
            if skipped < skip_rows:
                skipped += 1
                continue
            _bump_counter(doc_counters, counter_key)
            yield msgs


def _smoltalk2_blend_text_iter(
    config: str = 'Mid',
    *,
    skip_rows: int = 0,
    doc_counters: Optional[Dict[str, int]] = None,
    counter_key: str = 'smoltalk2_mid',
) -> Iterator[str]:
    """smoltalk2 conversations rendered to ChatML *text* for the pretrain blend."""
    for msgs in _smoltalk2_row_iter(
        config, skip_rows=skip_rows, doc_counters=doc_counters, counter_key=counter_key,
    ):
        text = format_chat_messages(msgs)
        if text.strip():
            yield text


# Format-aware source registry: single source of truth for schema + kind so the
# pretrain blend and the SFT stage agree on how each dataset is shaped.
#   schema: 'text' (raw web docs) | 'messages' (chat, rendered to ChatML text)
#   kind:   'web' (grammar/knowledge, warmup pool) | 'reason' | 'chat'
SOURCE_REGISTRY: Dict[str, Dict[str, str]] = {
    'dclm':          {'schema': 'text',     'kind': 'web',    'hf': 'HuggingFaceTB/dclm-edu'},
    'fineweb':       {'schema': 'text',     'kind': 'web',    'hf': 'HuggingFaceFW/fineweb-edu'},
    'smoltalk2_mid': {'schema': 'messages', 'kind': 'reason', 'hf': SMOLTALK2_REPO, 'config': 'Mid'},
    'smoltalk2_sft': {'schema': 'messages', 'kind': 'chat',   'hf': SMOLTALK2_REPO, 'config': 'SFT'},
}


def load_pretrain_holdout_val(
    tokenizer,
    seq_len: int = 2048,
    *,
    target_tokens: int = 500_000,
    edu_score_min: int = 3,
    holdout_pct: int = 5,
    use_cache: bool = True,
) -> TextDataset:
    """Cached in-distribution DCLM-Edu holdout (hash bucket), for pretrain val."""
    vocab_tag = len(tokenizer)
    cache_path = (
        Path('.cache') / 'v7_tokens'
        / f'pretrain_holdout_v{_PRETRAIN_HOLDOUT_CACHE_VERSION}_v{vocab_tag}'
        f'_t{target_tokens}_e{edu_score_min}_p{holdout_pct}.pt'
    )
    if use_cache and cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        tokens = cached['tokens']
        print(
            f"[cache] Pretrain holdout val from {cache_path} "
            f"({len(tokens):,} tokens)"
        )
        return TextDataset(tokens, seq_len)

    print(
        f"Building pretrain holdout val (~{holdout_pct}% hash bucket, "
        f"edu>={edu_score_min}, target>={target_tokens:,} tokens)..."
    )
    tokens: List[int] = []
    kept_docs = 0
    for text in _dclm_edu_text_iter(
        edu_score_min,
        only_holdout=True,
        holdout_pct=holdout_pct,
    ):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        ids.append(tokenizer.eos_token_id)
        tokens.extend(ids)
        kept_docs += 1
        if len(tokens) >= target_tokens:
            break

    if len(tokens) < target_tokens // 4:
        raise RuntimeError(
            f"Pretrain holdout too small ({len(tokens):,} tokens from {kept_docs} docs)"
        )

    out = torch.tensor(tokens[:target_tokens], dtype=torch.long)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'tokens': out,
            'kept_docs': kept_docs,
            'holdout_pct': holdout_pct,
            'edu_score_min': edu_score_min,
            'cache_version': _PRETRAIN_HOLDOUT_CACHE_VERSION,
        },
        cache_path,
    )
    print(f"  holdout val: {kept_docs:,} docs, {len(out):,} tokens -> {cache_path}")
    return TextDataset(out, seq_len)


def load_dclm_edu(
    seq_len: int = 2048,
    edu_score_min: int = 3,
    token_budget: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    holdout_pct: int = 5,
    exclude_holdout: bool = True,
):
    """Stream DCLM-Edu for pretrain; in-distro holdout val + WikiText secondary."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Streaming DCLM-Edu (edu_int_score>={edu_score_min}, "
        f"token_budget={token_budget or 'none'}, exclude_holdout={exclude_holdout})..."
    )
    train_ds = StreamingTokenChunkDataset(
        _dclm_edu_text_iter(
            edu_score_min,
            exclude_holdout=exclude_holdout,
            holdout_pct=holdout_pct,
        ),
        tokenizer,
        seq_len,
        max_tokens=token_budget,
        shuffle_buffer=10_000,
        default_source='dclm',
    )

    if exclude_holdout:
        val_ds = load_pretrain_holdout_val(
            tokenizer, seq_len=seq_len, edu_score_min=edu_score_min,
            holdout_pct=holdout_pct,
        )
    else:
        val_ds, _ = load_wikitext103_val(
            seq_len=seq_len,
            max_val_samples=max_val_samples,
        )
    return train_ds, val_ds, tokenizer


# ── Pretrain mix token cache (skip+tokenize once on CPU, reuse next round) ───

_PRETRAIN_CACHE_VERSION = 1
_PRETRAIN_CACHE_SHARD_ROWS = 50_000


def _pretrain_cache_meta(
    *,
    seq_len: int,
    token_budget: int,
    offset_tokens: int,
    edu_score_min: int,
    sources: Tuple[str, ...],
    weights: Optional[Tuple[float, ...]],
    fineweb_name: str,
    mix_seed: int,
    blend_warmup_tokens: int,
    skip_map: Dict[str, int],
) -> dict:
    return {
        'version': _PRETRAIN_CACHE_VERSION,
        'seq_len': seq_len,
        'token_budget': token_budget,
        'offset_tokens': offset_tokens,
        'edu_score_min': edu_score_min,
        'sources': list(sources),
        'weights': list(weights) if weights else None,
        'fineweb_name': fineweb_name,
        'mix_seed': mix_seed,
        'blend_warmup_tokens': blend_warmup_tokens,
        'skip_map': {k: int(v) for k, v in sorted(skip_map.items())},
        'chat_cache_version': _CHAT_CACHE_VERSION,
    }


def pretrain_mix_cache_dir(meta: dict, cache_root: Optional[Path] = None) -> Path:
    """Deterministic cache directory from blend parameters."""
    root = cache_root or (Path('.cache') / 'v7_tokens')
    blob = json.dumps(meta, sort_keys=True)
    tag = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return root / f'pretrain_mix_v{_PRETRAIN_CACHE_VERSION}_{tag}'


def _load_pretrain_mix_from_cache(cache_dir: Path):
    manifest = json.loads((cache_dir / 'manifest.json').read_text())
    shards = []
    for i in range(manifest['n_shards']):
        part = torch.load(cache_dir / f'train_{i:04d}.pt', weights_only=False)
        shards.append(StackedChunkDataset(part['input_ids'], part['labels']))
    train_ds = shards[0] if len(shards) == 1 else ConcatDataset(shards)
    train_ds.pretrain_cached = True  # type: ignore[attr-defined]
    print(
        f"[cache] Loaded pretrain mix from {cache_dir} "
        f"({manifest['n_rows']:,} rows, {manifest['token_budget']:,} tok, "
        f"offset={manifest['offset_tokens']:,}, "
        f"skips={manifest.get('start_docs', {})})"
    )
    return train_ds, manifest


def build_pretrain_mix_token_cache(
    *,
    seq_len: int = 2048,
    token_budget: int = 2_000_000_000,
    offset_tokens: int = 0,
    edu_score_min: int = 3,
    sources: Tuple[str, ...] = ('dclm', 'fineweb', 'smoltalk2_mid'),
    weights: Optional[Tuple[float, ...]] = None,
    fineweb_name: str = 'sample-10BT',
    mix_seed: int = 42,
    blend_warmup_tokens: int = 0,
    skip_map: Optional[Dict[str, int]] = None,
    cache_root: Optional[Path] = None,
    resume: bool = True,
) -> Path:
    """Tokenize a pretrain blend slice to sharded disk cache (CPU, resumable).

    ``offset_tokens`` skips the first N counted training tokens (same counter as
    ``StreamingTokenChunkDataset``) before writing -- use to prefetch the *next*
    round while the current round trains on GPU (offset = current TOKEN_BUDGET).
    """
    skip_map = dict(skip_map or {})
    meta = _pretrain_cache_meta(
        seq_len=seq_len,
        token_budget=token_budget,
        offset_tokens=offset_tokens,
        edu_score_min=edu_score_min,
        sources=sources,
        weights=weights,
        fineweb_name=fineweb_name,
        mix_seed=mix_seed,
        blend_warmup_tokens=blend_warmup_tokens,
        skip_map=skip_map,
    )
    cache_dir = pretrain_mix_cache_dir(meta, cache_root)
    if (cache_dir / 'manifest.json').exists():
        print(f"[cache] pretrain mix already complete: {cache_dir}")
        return cache_dir

    if chat_vocab := True:
        tokenizer = get_chat_tokenizer()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

    doc_counters: Dict[str, int] = {s: int(skip_map.get(s, 0)) for s in sources}
    start_docs = dict(doc_counters)
    token_counters: Dict[str, int] = {}

    tagged = []
    web_sources: List[str] = []
    for s in sources:
        spec = SOURCE_REGISTRY.get(s)
        if spec is None:
            raise ValueError(f"Unknown pretrain source: {s}")
        sk = skip_map.get(s, 0)
        if spec['kind'] == 'web':
            if s == 'dclm':
                it = _dclm_edu_text_iter(
                    edu_score_min, exclude_holdout=True,
                    holdout_pct=5, skip_docs=sk,
                    doc_counters=doc_counters, counter_key=s,
                )
            else:
                it = _fineweb_edu_text_iter(
                    edu_score_min, name=fineweb_name, exclude_holdout=True,
                    holdout_pct=5, skip_docs=sk,
                    doc_counters=doc_counters, counter_key=s,
                )
            web_sources.append(s)
        else:
            it = _smoltalk2_blend_text_iter(
                config=spec.get('config', 'Mid'), skip_rows=sk,
                doc_counters=doc_counters, counter_key=s,
            )
        tagged.append((s, it))

    if len(tagged) > 1:
        text_iter = _blend_interleave_text_iters(
            tagged, weights,
            warmup_tokens=blend_warmup_tokens, warmup_sources=tuple(web_sources),
            token_counters=token_counters, seed=mix_seed,
        )
    else:
        text_iter = _tag_source(tagged[0][0], tagged[0][1])

    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    n_rows = 0
    yielded_tokens = 0
    inp_buf: List[torch.Tensor] = []
    lab_buf: List[torch.Tensor] = []

    state_path = cache_dir / 'build_state.json'
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        shard_idx = int(state.get('shard_idx', 0))
        n_rows = int(state.get('n_rows', 0))
        yielded_tokens = int(state.get('yielded_tokens', 0))
        doc_counters.update(state.get('doc_counters', {}))
        token_counters.update(state.get('token_counters', {}))
        print(
            f"[cache] resuming pretrain build @ shard {shard_idx}, "
            f"rows={n_rows:,}, yielded_tok={yielded_tokens:,}",
            flush=True,
        )
        if yielded_tokens >= offset_tokens + token_budget:
            print(f"[cache] build already past target; finalizing manifest")
        else:
            # Must re-stream from start (skip is not seekable); resume only if
            # interrupted before any progress on this host.
            if shard_idx > 0 or yielded_tokens > 0:
                print(
                    "[cache] WARNING: partial shard resume requires re-streaming "
                    "from doc 0; delete cache_dir to rebuild cleanly.",
                    flush=True,
                )

    target_end = offset_tokens + token_budget
    print(
        f"[cache] Building pretrain mix -> {cache_dir}\n"
        f"  offset={offset_tokens:,} budget={token_budget:,} "
        f"skips={skip_map} sources={list(sources)}",
        flush=True,
    )

    def _flush_shard():
        nonlocal shard_idx, n_rows
        if not inp_buf:
            return
        inp = torch.stack(inp_buf)
        lab = torch.stack(lab_buf)
        path = cache_dir / f'train_{shard_idx:04d}.pt'
        _atomic_torch_save({'input_ids': inp, 'labels': lab}, path)
        n_rows += inp.shape[0]
        shard_idx += 1
        inp_buf.clear()
        lab_buf.clear()
        state_path.write_text(json.dumps({
            'shard_idx': shard_idx,
            'n_rows': n_rows,
            'yielded_tokens': yielded_tokens,
            'doc_counters': doc_counters,
            'token_counters': token_counters,
        }))
        print(
            f"  [cache] shard {shard_idx - 1:04d} rows={inp.shape[0]:,} "
            f"total_rows={n_rows:,} yielded_tok={yielded_tokens:,}",
            flush=True,
        )

    buffer: List[int] = []
    for item in text_iter:
        if isinstance(item, tuple):
            _, text = item
        else:
            text = item
        if not text or not str(text).strip():
            continue
        ids = tokenizer.encode(str(text), add_special_tokens=False)
        if not ids:
            continue
        ids.append(tokenizer.eos_token_id)
        buffer.extend(ids)
        while len(buffer) >= seq_len + 1:
            chunk = buffer[: seq_len + 1]
            buffer = buffer[seq_len + 1:]
            yielded_tokens += seq_len
            if yielded_tokens <= offset_tokens:
                continue
            if yielded_tokens > target_end:
                break
            inp_buf.append(torch.tensor(chunk[:-1], dtype=torch.long))
            lab_buf.append(torch.tensor(chunk[1:], dtype=torch.long))
            if len(inp_buf) >= _PRETRAIN_CACHE_SHARD_ROWS:
                _flush_shard()
        if yielded_tokens > target_end:
            break

    _flush_shard()
    manifest = {
        **meta,
        'n_shards': shard_idx,
        'n_rows': n_rows,
        'start_docs': start_docs,
        'end_docs': dict(doc_counters),
        'per_source_tokens': dict(token_counters),
    }
    (cache_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if state_path.exists():
        state_path.unlink()
    print(
        f"[cache] pretrain mix complete: {cache_dir} "
        f"({n_rows:,} rows, {min(yielded_tokens - offset_tokens, token_budget):,} tok written)",
        flush=True,
    )
    return cache_dir


def load_pretrain_mix(
    seq_len: int = 2048,
    edu_score_min: int = 3,
    token_budget: Optional[int] = None,
    sources: Tuple[str, ...] = ('dclm', 'fineweb'),
    weights: Optional[Tuple[float, ...]] = None,
    chat_vocab: bool = True,
    fineweb_name: str = 'sample-10BT',
    max_val_samples: Optional[int] = None,
    holdout_pct: int = 5,
    exclude_holdout: bool = True,
    mix_seed: int = 42,
    dclm_skip_docs: int = 0,
    fineweb_skip_docs: int = 0,
    skip_docs: Optional[Dict[str, int]] = None,
    blend_warmup_tokens: int = 0,
    token_counters: Optional[Dict[str, int]] = None,
    doc_counters: Optional[Dict[str, int]] = None,
    offset_tokens: int = 0,
    use_cache: bool = True,
):
    """Stream a blended corpus for from-scratch pretrain (knowledge+reasoning+chat).

    Format-aware via ``SOURCE_REGISTRY``: ``text`` sources (dclm/fineweb) stream
    raw web docs; ``messages`` sources (smoltalk2 Mid) are rendered to ChatML
    *text*. During the first ``blend_warmup_tokens`` tokens only web sources are
    drawn (grammar/knowledge), then the full weighted blend applies.

    Freshness across rounds: every source advances ``doc_counters[name]`` per
    yielded item; ``skip_docs[name]`` skips already-consumed items so no doc/row
    is reused. ``chat_vocab=True`` uses the ChatML+reasoning tokenizer (50261).
    """
    if chat_vocab:
        tokenizer = get_chat_tokenizer()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

    skip_map = dict(skip_docs or {})
    skip_map.setdefault('dclm', dclm_skip_docs)
    skip_map.setdefault('fineweb', fineweb_skip_docs)

    budget = token_budget or 0
    if use_cache and budget > 0:
        meta = _pretrain_cache_meta(
            seq_len=seq_len,
            token_budget=budget,
            offset_tokens=offset_tokens,
            edu_score_min=edu_score_min,
            sources=sources,
            weights=weights,
            fineweb_name=fineweb_name,
            mix_seed=mix_seed,
            blend_warmup_tokens=blend_warmup_tokens,
            skip_map=skip_map,
        )
        cache_dir = pretrain_mix_cache_dir(meta)
        if (cache_dir / 'manifest.json').exists():
            train_ds, manifest = _load_pretrain_mix_from_cache(cache_dir)
            if doc_counters is not None:
                for s in sources:
                    doc_counters[s] = int(manifest.get('end_docs', {}).get(s, skip_map.get(s, 0)))
            if token_counters is not None:
                token_counters.update(manifest.get('per_source_tokens', {}))
            if chat_vocab:
                tokenizer = get_chat_tokenizer()
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained('gpt2')
                tokenizer.pad_token = tokenizer.eos_token
            if exclude_holdout:
                val_ds = load_pretrain_holdout_val(
                    tokenizer, seq_len=seq_len, edu_score_min=edu_score_min,
                    holdout_pct=holdout_pct,
                )
            else:
                val_ds, _ = load_wikitext103_val(
                    seq_len=seq_len, max_val_samples=max_val_samples,
                )
            return train_ds, val_ds, tokenizer

    # Seed cursors at prior cumulative so per_source_docs stays cumulative.
    if doc_counters is not None:
        for s in sources:
            doc_counters.setdefault(s, skip_map.get(s, 0))

    tagged = []
    web_sources: List[str] = []
    for s in sources:
        spec = SOURCE_REGISTRY.get(s)
        if spec is None:
            raise ValueError(f"Unknown pretrain source: {s} (known: {list(SOURCE_REGISTRY)})")
        sk = skip_map.get(s, 0)
        if spec['kind'] == 'web':
            if s == 'dclm':
                it = _dclm_edu_text_iter(
                    edu_score_min, exclude_holdout=exclude_holdout,
                    holdout_pct=holdout_pct, skip_docs=sk,
                    doc_counters=doc_counters, counter_key=s,
                )
            else:  # fineweb
                it = _fineweb_edu_text_iter(
                    edu_score_min, name=fineweb_name, exclude_holdout=exclude_holdout,
                    holdout_pct=holdout_pct, skip_docs=sk,
                    doc_counters=doc_counters, counter_key=s,
                )
            web_sources.append(s)
        else:  # messages -> ChatML text
            it = _smoltalk2_blend_text_iter(
                config=spec.get('config', 'Mid'), skip_rows=sk,
                doc_counters=doc_counters, counter_key=s,
            )
        tagged.append((s, it))

    print(
        f"Streaming pretrain blend: sources={list(sources)} "
        f"weights={list(weights) if weights else 'uniform'} edu>={edu_score_min} "
        f"budget={token_budget or 'none'} vocab={len(tokenizer)} "
        f"holdout_pct={holdout_pct} exclude_holdout={exclude_holdout} "
        f"mix_seed={mix_seed} warmup={blend_warmup_tokens:,} "
        f"skips={ {s: skip_map.get(s, 0) for s in sources} }"
    )
    if len(tagged) > 1:
        text_iter = _blend_interleave_text_iters(
            tagged, weights,
            warmup_tokens=blend_warmup_tokens, warmup_sources=tuple(web_sources),
            token_counters=token_counters, seed=mix_seed,
        )
    else:
        text_iter = _tag_source(tagged[0][0], tagged[0][1])
    train_ds = StreamingTokenChunkDataset(
        text_iter,
        tokenizer,
        seq_len,
        max_tokens=token_budget,
        shuffle_buffer=10_000,
        token_counters=token_counters,
    )
    if exclude_holdout:
        val_ds = load_pretrain_holdout_val(
            tokenizer, seq_len=seq_len, edu_score_min=edu_score_min,
            holdout_pct=holdout_pct,
        )
    else:
        val_ds, _ = load_wikitext103_val(
            seq_len=seq_len, max_val_samples=max_val_samples,
        )
    return train_ds, val_ds, tokenizer


def load_fineweb_edu(
    seq_len: int = 2048,
    edu_score_min: int = 3,
    token_budget: Optional[int] = None,
    chat_vocab: bool = False,
    fineweb_name: str = 'sample-10BT',
    max_val_samples: Optional[int] = None,
    holdout_pct: int = 5,
    dclm_skip_docs: int = 0,
    fineweb_skip_docs: int = 0,
    mix_seed: int = 42,
    token_counters: Optional[Dict[str, int]] = None,
):
    """Stream FineWeb-Edu only (mirror of load_dclm_edu)."""
    return load_pretrain_mix(
        seq_len=seq_len, edu_score_min=edu_score_min, token_budget=token_budget,
        sources=('fineweb',), chat_vocab=chat_vocab, fineweb_name=fineweb_name,
        max_val_samples=max_val_samples, holdout_pct=holdout_pct,
        dclm_skip_docs=dclm_skip_docs, fineweb_skip_docs=fineweb_skip_docs,
        mix_seed=mix_seed, token_counters=token_counters,
    )


def _think_keep(messages: List[dict], think_fraction: float) -> bool:
    """Deterministically keep a fraction of think (reasoning) conversations.

    no_think conversations are always kept. think conversations are kept with
    probability ~``think_fraction`` via a stable content hash, so a 100M model
    trains mostly on direct answers with a capped dose of reasoning traces.
    """
    if not _messages_have_think(messages):
        return True
    if think_fraction >= 1.0:
        return True
    if think_fraction <= 0.0:
        return False
    blob = json.dumps(messages, ensure_ascii=False)[:4096]
    return (zlib.adler32(blob.encode('utf-8', errors='ignore')) % 1000) < int(think_fraction * 1000)


def load_smoltalk2(
    seq_len: int = 2048,
    max_samples: Optional[int] = None,
    sft_filter: str = 'hard',
    think_fraction: float = 0.15,
    smoltalk2_skip_rows: int = 0,
    use_cache: bool = True,
    **_unused,
):
    """Real smoltalk2 SFT-config chat fine-tuning (ChatML, assistant-only loss).

    Uses ``HuggingFaceTB/smoltalk2`` config ``SFT`` (chat + reasoning, think/no_think).
    ``think_fraction`` caps the share of reasoning conversations so a small base
    stays mostly on direct answers. Disjoint from the pretrain blend, which draws
    reasoning from the ``Mid`` config. In-distribution holdout val (not WikiText).
    """
    def _rows():
        n = 0
        for msgs in _smoltalk2_row_iter('SFT', skip_rows=smoltalk2_skip_rows):
            if not _think_keep(msgs, think_fraction):
                continue
            yield {'messages': msgs}
            n += 1
            if max_samples and n >= max_samples:
                break

    tag = f"smoltalk2_sft_tf{int(think_fraction * 100)}"
    return load_chat_sft(
        tag, _rows,
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
