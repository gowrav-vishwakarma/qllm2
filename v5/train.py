"""
V5 Training Script.

Matches the reviewer's benchmark: 20k TinyStories, same tokenizer (GPT-2),
same optimizer (AdamW), same schedule (cosine), 20 epochs, small scale.

Usage:
    python -m v5.train --size small --epochs 20 --max_samples 20000
    python -m v5.train --size small-matched --epochs 20  # match ~8M params
    python -m v5.train --resume checkpoints_v5/best_model.pt  # resume training
"""

import os
import sys
import time
import math
import argparse
import hashlib
import json
import re
import unicodedata
import warnings
from contextlib import contextmanager, nullcontext
from array import array
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from v5.model import AlgebraicLM, create_model, ModelOutput
from v5.config import V5Config, get_config
from v5.init import list_strategies


# Suppress a known Inductor warning that is performance-only and noisy during
# compiled training. Keep this narrow so other warnings still surface.
warnings.filterwarnings(
    "ignore",
    message=r".*Online softmax is disabled on the fly since Inductor decides to.*",
    category=UserWarning,
    module=r"torch\._inductor\.lowering",
)


# ---------------------------------------------------------------------------
# TeeLogger -- duplicate stdout to log file with wall-time prefix
# ---------------------------------------------------------------------------

class TeeLogger:
    """Writes to both stdout and a log file. Each log line gets a timestamp."""

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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Pre-tokenized text dataset for language modeling."""

    def __init__(self, tokens: torch.Tensor, seq_len: int = 512):
        self.seq_len = seq_len
        n_chunks = len(tokens) // (seq_len + 1)
        self.data = tokens[:n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.data[idx]
        return {
            'input_ids': chunk[:-1],
            'labels': chunk[1:],
        }


MOJIBAKE_MARKERS = (
    'â€', 'â€œ', 'â€\x9d', 'â€™', 'â€˜', 'â€“', 'â€”',
    'Ã', 'Â', 'ðŸ', '\ufffd',
)
EXCESS_BLANK_LINES_RE = re.compile(r'\n{3,}')
TOKEN_CACHE_VERSION = 1
MOJIBAKE_REPLACEMENTS = (
    ('â€œ', '“'),
    ('â€\x9d', '”'),
    ('â€˜', '‘'),
    ('â€™', '’'),
    ('â€“', '–'),
    ('â€”', '—'),
    ('â€¦', '…'),
    ('Â¡', '¡'),
    ('Â¿', '¿'),
    ('â€', '”'),
    ('Â', ''),
)


@dataclass
class TextPrepStats:
    total_texts: int = 0
    kept_texts: int = 0
    repaired_texts: int = 0
    suspicious_markers_before: int = 0
    suspicious_markers_after: int = 0
    total_tokens: int = 0


def build_token_cache_meta(
    split_name: str,
    max_samples: Optional[int],
    tokenizer_name: str,
    repair_text: bool,
) -> Dict[str, object]:
    return {
        'version': TOKEN_CACHE_VERSION,
        'dataset': 'roneneldan/TinyStories',
        'split': split_name,
        'max_samples': max_samples,
        'tokenizer': tokenizer_name,
        'repair_text': repair_text,
    }


def token_cache_path(cache_dir: str, meta: Dict[str, object]) -> Path:
    payload = json.dumps(meta, sort_keys=True, separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return Path(cache_dir) / f"{meta['split']}_tokens_v{TOKEN_CACHE_VERSION}_{digest}.pt"


def load_token_cache(path: Path, expected_meta: Dict[str, object]) -> Optional[tuple[torch.Tensor, TextPrepStats]]:
    if not path.exists():
        return None

    print(f"Loading token cache from {path}")
    try:
        cache = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Token cache load failed: {e}. Rebuilding...")
        return None

    if cache.get('meta') != expected_meta:
        print("Token cache metadata mismatch. Rebuilding...")
        return None

    tokens = cache.get('tokens')
    if not isinstance(tokens, torch.Tensor):
        print("Token cache missing tensor payload. Rebuilding...")
        return None

    tokens = tokens.to(torch.long)
    stats = TextPrepStats(**cache.get('stats', {}))
    print(
        f"Loaded cached {expected_meta['split']} tokens: {tokens.numel():,} "
        f"| repaired {stats.repaired_texts:,}/{stats.kept_texts:,} stories"
    )
    return tokens, stats


def save_token_cache(
    path: Path,
    meta: Dict[str, object],
    tokens: torch.Tensor,
    stats: TextPrepStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        'meta': meta,
        'tokens': tokens.cpu(),
        'stats': asdict(stats),
    }
    torch.save(cache, path)
    print(f"Saved token cache to {path}")


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def normalize_story_text(text: str, repair_text: bool = True) -> tuple[str, bool, int, int]:
    """Normalize story formatting and repair common UTF-8/Latin-1 mojibake."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\ufeff', '').replace('\xa0', ' ')
    text = unicodedata.normalize('NFC', text)

    score_before = _mojibake_score(text)
    repaired = False

    if repair_text and score_before > 0:
        best = text
        best_score = score_before
        for codec in ('cp1252', 'latin1'):
            candidate_best = best
            candidate_score = best_score
            for _ in range(2):
                try:
                    candidate = candidate_best.encode(codec).decode('utf-8')
                except UnicodeError:
                    break
                candidate = unicodedata.normalize('NFC', candidate)
                new_score = _mojibake_score(candidate)
                if new_score < candidate_score:
                    candidate_best = candidate
                    candidate_score = new_score
                else:
                    break
            if candidate_score < best_score:
                best = candidate_best
                best_score = candidate_score
                repaired = True

        fallback = best
        for bad, good in MOJIBAKE_REPLACEMENTS:
            fallback = fallback.replace(bad, good)
        fallback_score = _mojibake_score(fallback)
        if fallback_score < best_score:
            best = fallback
            best_score = fallback_score
            repaired = True
        text = best

    text = '\n'.join(line.rstrip() for line in text.splitlines())
    text = EXCESS_BLANK_LINES_RE.sub('\n\n', text)
    text = text.strip()
    score_after = _mojibake_score(text)
    return text, repaired, score_before, score_after


def resolve_max_val_samples(
    max_samples: Optional[int],
    max_val_samples: Optional[int],
) -> Optional[int]:
    if max_val_samples is not None:
        return max_val_samples
    if max_samples is None:
        return None
    return max(1000, max_samples // 10)


def tokenize_split(
    ds,
    tokenizer,
    split_name: str,
    max_samples: Optional[int],
    tokenize_batch_size: int,
    repair_text: bool,
    cache_dir: Optional[str],
    use_cache: bool,
) -> tuple[torch.Tensor, TextPrepStats]:
    """Clean and tokenize one TinyStories split in batches."""
    limit = len(ds) if max_samples is None else min(max_samples, len(ds))
    eos_token_id = tokenizer.eos_token_id
    token_buffer = array('I')
    stats = TextPrepStats()
    tokenizer_name = getattr(tokenizer, 'name_or_path', tokenizer.__class__.__name__)
    cache_meta = build_token_cache_meta(
        split_name=split_name,
        max_samples=limit,
        tokenizer_name=tokenizer_name,
        repair_text=repair_text,
    )
    cache_path = token_cache_path(cache_dir, cache_meta) if (use_cache and cache_dir) else None

    if cache_path is not None:
        cached = load_token_cache(cache_path, cache_meta)
        if cached is not None:
            return cached

    print(
        f"Preparing {split_name} split: {limit:,} stories "
        f"(tokenize_batch_size={tokenize_batch_size}, text_repair={repair_text})..."
    )

    for start in range(0, limit, tokenize_batch_size):
        end = min(start + tokenize_batch_size, limit)
        batch = ds[start:end]
        batch_texts = []

        for text in batch['text']:
            stats.total_texts += 1
            cleaned, repaired, score_before, score_after = normalize_story_text(
                text, repair_text=repair_text
            )
            stats.suspicious_markers_before += score_before
            stats.suspicious_markers_after += score_after
            if repaired:
                stats.repaired_texts += 1
            if not cleaned:
                continue
            stats.kept_texts += 1
            batch_texts.append(cleaned)

        if not batch_texts:
            continue

        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )
        for ids in encoded['input_ids']:
            token_buffer.extend(ids)
            token_buffer.append(eos_token_id)
            stats.total_tokens += len(ids) + 1

    tokens = torch.tensor(token_buffer, dtype=torch.long)
    print(
        f"{split_name.capitalize()} stats: kept {stats.kept_texts:,}/{stats.total_texts:,} "
        f"stories | repaired {stats.repaired_texts:,} | mojibake markers "
        f"{stats.suspicious_markers_before:,} -> {stats.suspicious_markers_after:,} | "
        f"tokens {stats.total_tokens:,}"
    )
    if cache_path is not None:
        save_token_cache(cache_path, cache_meta, tokens, stats)
    return tokens, stats


def load_tinystories(
    max_samples: Optional[int] = 20000,
    seq_len: int = 512,
    max_val_samples: Optional[int] = None,
    tokenize_batch_size: int = 512,
    repair_text: bool = True,
    use_cache: bool = True,
    cache_dir: str = '.cache/v5_tokens',
):
    """Load TinyStories, clean text, and tokenize with the GPT-2 tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    try:
        from datasets import load_dataset
        resolved_max_val = resolve_max_val_samples(max_samples, max_val_samples)
        print(
            f"Loading TinyStories (train_max_samples={max_samples}, "
            f"val_max_samples={resolved_max_val})..."
        )
        train_split = load_dataset('roneneldan/TinyStories', split='train')
        val_split = load_dataset('roneneldan/TinyStories', split='validation')
    except Exception as e:
        print(f"Failed to load TinyStories: {e}")
        print("Using random data as fallback.")
        return _random_dataset(50257, seq_len, 1000), _random_dataset(50257, seq_len, 100), tokenizer

    train_tokens, _ = tokenize_split(
        train_split,
        tokenizer,
        split_name='train',
        max_samples=max_samples,
        tokenize_batch_size=tokenize_batch_size,
        repair_text=repair_text,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    val_tokens, _ = tokenize_split(
        val_split,
        tokenizer,
        split_name='validation',
        max_samples=resolved_max_val,
        tokenize_batch_size=tokenize_batch_size,
        repair_text=repair_text,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)

    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
    return train_ds, val_ds, tokenizer


def _random_dataset(vocab_size, seq_len, num_samples):
    tokens = torch.randint(1, vocab_size, (num_samples * (seq_len + 1),))
    return TextDataset(tokens, seq_len)


def resolve_amp_dtype(device: torch.device, requested: str) -> Optional[torch.dtype]:
    """Pick a CUDA autocast dtype that matches the local PyTorch/CUDA build."""
    if device.type != 'cuda':
        return None

    bf16_supported = False
    try:
        bf16_supported = torch.cuda.is_bf16_supported()
    except Exception:
        bf16_supported = False

    if requested == 'bf16':
        if bf16_supported:
            return torch.bfloat16
        print("Requested bf16 AMP, but this CUDA stack reports no bf16 support. Falling back to fp16.")
        return torch.float16

    if requested == 'fp16':
        return torch.float16

    return torch.bfloat16 if bf16_supported else torch.float16


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: V5Config,
    total_steps: int,
):
    total_steps = max(total_steps, 1)
    if config.lr_schedule == 'cosine' or config.warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
        )

    if config.lr_schedule == 'warmup_cosine':
        warmup_steps = min(config.warmup_steps, max(total_steps - 1, 0))
        if warmup_steps <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
            )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=max(1.0 / warmup_steps, 1e-6),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: AlgebraicLM,
        config: V5Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer=None,
        checkpoint_dir: str = 'checkpoints_v5',
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

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        if self.device.type == 'cuda':
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
            torch.backends.cudnn.allow_tf32 = config.allow_tf32

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        total_steps = config.max_epochs * len(train_loader)
        self.scheduler = build_scheduler(self.optimizer, config, total_steps)

        self.use_amp = self.device.type == 'cuda'
        self.amp_dtype = resolve_amp_dtype(self.device, config.amp_dtype)
        self.use_grad_scaler = self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler('cuda') if self.use_grad_scaler else None
        self.use_non_blocking = self.device.type == 'cuda'
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_ppl = float('inf')

    def _autocast_context(self):
        autocast_kwargs = {
            'device_type': self.device.type,
            'enabled': self.use_amp,
        }
        if self.amp_dtype is not None:
            autocast_kwargs['dtype'] = self.amp_dtype
        return torch.amp.autocast(**autocast_kwargs)

    def _amp_dtype_name(self) -> str:
        if self.amp_dtype == torch.bfloat16:
            return 'bf16'
        if self.amp_dtype == torch.float16:
            return 'fp16'
        return 'disabled'

    def _cuda_memory_summary(self) -> str:
        if self.device.type != 'cuda':
            return ''
        gib = 1024 ** 3
        allocated = torch.cuda.memory_allocated(self.device) / gib
        reserved = torch.cuda.memory_reserved(self.device) / gib
        peak_reserved = torch.cuda.max_memory_reserved(self.device) / gib
        return f"mem {allocated:.1f}/{reserved:.1f}G peak {peak_reserved:.1f}G"

    def _model_base(self):
        return self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

    @contextmanager
    def _temporary_attention_backend(self, backend: str):
        base_model = self._model_base()
        old = {}
        if hasattr(base_model, 'attn_layers'):
            for key, layer in base_model.attn_layers.items():
                old[key] = layer.attention_backend
                layer.attention_backend = backend
        try:
            yield
        finally:
            if hasattr(base_model, 'attn_layers'):
                for key, layer in base_model.attn_layers.items():
                    layer.attention_backend = old.get(key, layer.attention_backend)

    def _compute_val_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        use_autocast: bool,
        force_native_attention: bool,
    ) -> torch.Tensor:
        model = self._model_base() if force_native_attention else self.model
        autocast_enabled = use_autocast and self.use_amp and not force_native_attention
        context = self._autocast_context() if autocast_enabled else nullcontext()

        if force_native_attention:
            with self._temporary_attention_backend('native'):
                with context:
                    output = model(input_ids)
                    logits = output.logits.view(-1, output.logits.size(-1))
                    return F.cross_entropy(logits, labels.view(-1))

        with context:
            output = model(input_ids)
            logits = output.logits.view(-1, output.logits.size(-1))
            return F.cross_entropy(logits, labels.view(-1))

    def _validate_batch_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        batch_idx: int,
    ) -> Optional[torch.Tensor]:
        use_autocast = self.config.attention_backend == 'native'
        loss = self._compute_val_loss(
            input_ids,
            labels,
            use_autocast=use_autocast,
            force_native_attention=False,
        )
        if torch.isfinite(loss):
            return loss

        if use_autocast:
            print(
                f"Validation batch {batch_idx} produced non-finite loss under normal eval; "
                "retrying in full precision."
            )
            loss = self._compute_val_loss(
                input_ids,
                labels,
                use_autocast=False,
                force_native_attention=False,
            )
            if torch.isfinite(loss):
                return loss
        else:
            print(
                f"Validation batch {batch_idx} produced non-finite loss in full precision eval; "
                "retrying with native attention backend."
            )

        if self.config.attention_backend != 'native':
            if use_autocast:
                print(
                    f"Validation batch {batch_idx} is still non-finite; "
                    "retrying with native attention backend."
                )
            loss = self._compute_val_loss(
                input_ids,
                labels,
                use_autocast=False,
                force_native_attention=True,
            )
            if torch.isfinite(loss):
                return loss

        print(f"Validation batch {batch_idx} remained non-finite after all retries; skipping it.")
        return None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_div_loss_raw = 0.0
        total_div_loss_weighted = 0.0
        num_batches = 0
        epoch_start = time.time()
        first_step_start = None
        last_log_time = epoch_start
        last_log_batch = 0
        num_total_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx == 0:
                first_step_start = time.time()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

            input_ids = batch['input_ids'].to(self.device, non_blocking=self.use_non_blocking)
            labels = batch['labels'].to(self.device, non_blocking=self.use_non_blocking)
            seq_len = input_ids.shape[1]

            with self._autocast_context():
                output = self.model(input_ids)

                logits = output.logits.view(-1, output.logits.size(-1))
                ce_loss = F.cross_entropy(logits, labels.view(-1))

                loss = ce_loss
                div_loss_raw_val = 0.0
                div_loss_weighted_val = 0.0
                if output.diversity_loss is not None:
                    div_loss = output.diversity_loss * self.config.diversity_loss_weight
                    loss = loss + div_loss
                    div_loss_raw_val = output.diversity_loss.item()
                    div_loss_weighted_val = div_loss.item()

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
            total_div_loss_raw += div_loss_raw_val
            total_div_loss_weighted += div_loss_weighted_val
            num_batches += 1

            if self.verbose and batch_idx == 0 and first_step_start is not None:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                first_step_s = time.time() - first_step_start
                compile_note = " (includes compile)" if self.config.compile_model else ""
                print(f"  First step wall time: {first_step_s:.1f}s{compile_note}")

            should_log = self.verbose and (
                batch_idx == 0 or
                (batch_idx + 1) % self.config.log_interval == 0 or
                batch_idx == num_total_batches - 1
            )
            if should_log:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                ppl = math.exp(min(ce_loss.item(), 20))
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                current_batch = batch_idx + 1
                avg_samples_per_sec = current_batch * self.config.batch_size / elapsed
                avg_tokens_per_sec = avg_samples_per_sec * seq_len
                interval_batches = current_batch - last_log_batch
                interval_elapsed = max(time.time() - last_log_time, 1e-6)
                inst_samples_per_sec = interval_batches * self.config.batch_size / interval_elapsed
                inst_tokens_per_sec = inst_samples_per_sec * seq_len
                progress = 100.0 * current_batch / num_total_batches
                batches_left = num_total_batches - current_batch
                eta_seconds = batches_left * (interval_elapsed / max(interval_batches, 1))
                mem_summary = self._cuda_memory_summary()
                print(
                    f"  [{epoch+1}] batch {batch_idx}/{num_total_batches} "
                    f"loss={ce_loss.item():.4f} ppl={ppl:.1f} "
                    f"div={div_loss_raw_val:.2e} wdiv={div_loss_weighted_val:.2e} lr={lr:.2e} "
                    f"| inst {inst_tokens_per_sec:.0f} tok/s avg {avg_tokens_per_sec:.0f} tok/s "
                    f"| {progress:5.1f}% eta {format_duration(eta_seconds)}"
                    f"{' | ' + mem_summary if mem_summary else ''}"
                )
                last_log_time = time.time()
                last_log_batch = current_batch

        return {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'div_loss_raw': total_div_loss_raw / num_batches,
            'div_loss_weighted': total_div_loss_weighted / num_batches,
            'ppl': math.exp(min(total_ce_loss / num_batches, 20)),
            'avg_tokens_per_sec': (
                (num_batches * self.config.batch_size * seq_len) /
                max(time.time() - epoch_start, 1e-6)
            ),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        skipped_batches = 0

        for batch_idx, batch in enumerate(self.val_loader):
            input_ids = batch['input_ids'].to(self.device, non_blocking=self.use_non_blocking)
            labels = batch['labels'].to(self.device, non_blocking=self.use_non_blocking)

            loss = self._validate_batch_loss(input_ids, labels, batch_idx)
            if loss is None:
                skipped_batches += 1
                continue

            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            print("Validation failed: all validation batches were non-finite.")
            return {
                'val_loss': float('nan'),
                'val_ppl': float('nan'),
            }

        if skipped_batches > 0:
            print(f"Validation skipped {skipped_batches} non-finite batch(es) after retries.")

        avg_loss = total_loss / num_batches
        return {
            'val_loss': avg_loss,
            'val_ppl': math.exp(min(avg_loss, 20)),
        }

    @torch.no_grad()
    def generate_sample(self, prompt: str = "The quick brown", max_tokens: int = 100) -> str:
        """Generate sample text for quality check."""
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
        if self.device.type == 'cuda':
            print(f"AMP: {self._amp_dtype_name()} | TF32: {self.config.allow_tf32}")
        params = self.model.count_parameters() if hasattr(self.model, 'count_parameters') else {}
        if params:
            print(f"Parameters: {params}")
            print(f"Total: {params['total']:,} ({params['total']/1e6:.1f}M)")
        print(f"Epochs: {self.start_epoch+1}..{self.config.max_epochs}, Batches/epoch: {len(self.train_loader)}")
        print()

        for epoch in range(self.start_epoch, self.config.max_epochs):
            self._current_epoch = epoch

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
                f"| Div: {train_metrics['div_loss_raw']:.2e} "
                f"(w {train_metrics['div_loss_weighted']:.2e}) "
                f"| Tok/s: {train_metrics['avg_tokens_per_sec']:.0f} "
                f"| "
                f"Time: {epoch_time:.1f}s"
            )

            is_best = False
            if self.val_loader is not None:
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

            # Save an epoch-numbered snapshot every epoch, regardless of val
            if self.save_checkpoints:
                self.save_checkpoint(f'epoch-{epoch+1}.pth')

            # Save best checkpoint
            if self.save_checkpoints and is_best:
                self.save_checkpoint('best_model.pt')

            # Periodic checkpoint every 5 epochs
            if self.save_checkpoints and (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

            # Generate sample after each epoch
            if self.tokenizer is not None:
                try:
                    text = self.generate_sample("The quick brown")
                    print(f"\nPrompt: The quick brown")
                    print(f"Generated: {text}")
                except Exception as e:
                    print(f"(Sample generation failed: {e})")

        # Final checkpoint
        self._current_epoch = self.config.max_epochs - 1
        if self.save_checkpoints:
            self.save_checkpoint('final_model.pt')

        total_time = time.time() - training_start
        print(f"\nTraining complete!")
        print(f"Total wall time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        print(f"Best Val Loss: {self.best_val_loss:.4f}, Best Val PPL: {self.best_val_ppl:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train V5 Algebraic LM')
    parser.add_argument('--size', type=str, default='small',
                        choices=['tiny', 'small', 'small-matched', 'medium', 'large'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--lr_schedule', type=str, default=None,
                        choices=['cosine', 'warmup_cosine'],
                        help='Learning-rate schedule')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Warmup steps for supported LR schedules')
    parser.add_argument('--max_samples', type=int, default=20000)
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Max TinyStories validation samples to load')
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--tokenize_batch_size', type=int, default=512,
                        help='Texts per batch during preprocessing tokenization')
    parser.add_argument('--cache_dir', type=str, default='.cache/v5_tokens',
                        help='Token cache directory')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable token cache for TinyStories preprocessing')
    parser.add_argument('--log_interval', type=int, default=None,
                        help='Batch interval for progress logging')
    parser.add_argument('--num_banks', type=int, default=None)
    parser.add_argument('--window_size', type=int, default=None,
                        help='Sliding attention window size')
    parser.add_argument('--no_text_repair', action='store_true',
                        help='Disable mojibake repair and text normalization')
    parser.add_argument('--no_attention', action='store_true')
    parser.add_argument('--no_banks', action='store_true')
    parser.add_argument('--init_strategy', type=str, default=None,
                        choices=list_strategies(),
                        help='Structured initialization strategy (default: from config)')
    parser.add_argument('--init_seed', type=int, default=None,
                        help='Seed for init (auto-generated if not set)')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v5')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--amp_dtype', type=str, default=None,
                        choices=['auto', 'bf16', 'fp16'],
                        help='CUDA autocast dtype (default: auto)')
    parser.add_argument('--no_tf32', action='store_true',
                        help='Disable TF32 matmul/cuDNN acceleration on CUDA')
    parser.add_argument('--attention_backend', type=str, default=None,
                        choices=['native', 'xformers', 'auto'],
                        help='Attention backend (default: native)')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile for speedup')
    parser.add_argument('--compile_mode', type=str, default=None,
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--fullgraph', action='store_true',
                        help='Enable fullgraph=True for torch.compile')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers on CUDA (default: from config)')
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Disable pin_memory on CUDA DataLoaders')
    args = parser.parse_args()

    config = get_config(args.size)
    config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.lr_schedule is not None:
        config.lr_schedule = args.lr_schedule
    if args.warmup_steps is not None:
        config.warmup_steps = max(0, args.warmup_steps)
    if args.num_banks is not None:
        config.num_banks = args.num_banks
    if args.window_size is not None:
        config.window_size = args.window_size
    if args.no_attention:
        config.attn_every_k = 0
    if args.no_banks:
        config.num_banks = 0
    if args.init_strategy is not None:
        config.init_strategy = args.init_strategy
    if args.amp_dtype is not None:
        config.amp_dtype = args.amp_dtype
    if args.no_tf32:
        config.allow_tf32 = False
    if args.attention_backend is not None:
        config.attention_backend = args.attention_backend
    if args.compile:
        config.compile_model = True
    if args.compile_mode is not None:
        config.compile_mode = args.compile_mode
    if args.fullgraph:
        config.compile_fullgraph = True
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.no_pin_memory:
        config.pin_memory = False
    if args.log_interval is not None:
        config.log_interval = max(1, args.log_interval)
    config.init_seed = args.init_seed

    # Set up TeeLogger: append on resume, overwrite on fresh start
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'v5_train_{args.size}.log'
    log_mode = 'a' if args.resume else 'w'
    tee = TeeLogger(log_path, mode=log_mode)
    sys.stdout = tee

    if args.resume:
        print(f"\n--- Resumed from {args.resume} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    print(f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("V5 Algebraic Language Model")
    print("=" * 60)
    print(f"Size: {args.size}")
    print(f"Complex dim: {config.dim} (= {config.dim * 2} real values/position)")
    print(f"SSM state dim: {config.state_dim}")
    print(f"Layers: {config.num_layers}")
    print(f"Banks: {config.num_banks}")
    print(f"Attention every: {config.attn_every_k} layers (0=none)")
    print(f"Attention window: {config.window_size}")
    print(f"Epochs: {config.max_epochs}")
    print(f"Max samples: {args.max_samples}")
    print(f"Max val samples: {resolve_max_val_samples(args.max_samples, args.max_val_samples)}")
    print(f"LR: {config.learning_rate:.2e} | weight_decay: {config.weight_decay:.3f}")
    print(f"Dropout: {config.dropout:.3f}")
    print(f"LR schedule: {config.lr_schedule} | warmup_steps: {config.warmup_steps}")
    print(f"Tokenize batch size: {args.tokenize_batch_size}")
    print(f"Text repair: {not args.no_text_repair}")
    print(f"Token cache: {not args.no_cache} (dir={args.cache_dir})")
    print(f"Log interval: {config.log_interval}")
    print(f"AMP dtype: {config.amp_dtype}")
    print(f"TF32 enabled: {config.allow_tf32}")
    print(f"Attention backend: {config.attention_backend}")
    validation_mode = "autocast" if config.attention_backend == 'native' else "full precision"
    print(f"Validation mode: {validation_mode}")
    print(
        f"Compile: {config.compile_model} "
        f"(mode={config.compile_mode}, fullgraph={config.compile_fullgraph})"
    )
    print(f"DataLoader workers: {config.num_workers}, pin_memory: {config.pin_memory}")
    print(f"Log file: {log_path}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 60)

    # Load data
    train_ds, val_ds, tokenizer = load_tinystories(
        args.max_samples,
        args.seq_len,
        max_val_samples=args.max_val_samples,
        tokenize_batch_size=args.tokenize_batch_size,
        repair_text=not args.no_text_repair,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
    )

    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = args.seq_len

    use_cuda = torch.cuda.is_available()
    num_workers = config.num_workers if use_cuda else 0
    pin_memory = config.pin_memory and use_cuda
    loader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    # Create model (resolves init_seed and stores in config)
    model = create_model(config)
    init_info = model.initializer_info
    print(f"Init strategy: {init_info['init_strategy']} (seed: {init_info['init_seed']})")

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_ppl = checkpoint.get('best_val_ppl', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        if 'init_strategy' in checkpoint and 'init_seed' in checkpoint:
            print(f"Original init: {checkpoint['init_strategy']} (seed: {checkpoint['init_seed']})")

    if config.compile_model:
        if hasattr(torch, 'compile'):
            print(
                f"Compiling model with torch.compile "
                f"(mode='{config.compile_mode}', fullgraph={config.compile_fullgraph})..."
            )
            try:
                model = torch.compile(
                    model,
                    mode=config.compile_mode,
                    fullgraph=config.compile_fullgraph,
                )
                print("Model compiled successfully.")
            except Exception as e:
                print(f"Compilation failed: {e}")
                print("Continuing without torch.compile.")
                config.compile_model = False
        else:
            print("torch.compile is unavailable in this PyTorch build. Continuing without compilation.")
            config.compile_model = False

    # Train
    trainer = Trainer(
        model, config, train_loader, val_loader,
        tokenizer=tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        start_epoch=start_epoch,
    )

    # Restore optimizer/scheduler state on resume
    if args.resume and 'optimizer_state_dict' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.best_val_loss = best_val_loss
        trainer.best_val_ppl = best_val_ppl

    trainer.train()

    print(f"\nWall clock end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Restore stdout and close log
    sys.stdout = tee._stdout
    tee.close()


if __name__ == '__main__':
    main()
