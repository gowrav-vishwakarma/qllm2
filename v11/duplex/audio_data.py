"""Stage 1 audio dataset (open fallback; Fisher LDC when available)."""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import Dataset

from v11.duplex.interleave import ScenarioKind, block_to_token_lists, build_block
from v11.duplex.thinking import VOCAB

# Stage 1 data sources (see plan):
#   Primary (needs LDC license): Fisher English LDC97S62 dual-channel ~2000h
#   Fallback (open): LibriSpeech clean train.100 via Parquet (no HF loading script)
#   Multilingual: ai4bharat/Kathbath (Indian langs, audio bytes in parquet)
DEFAULT_STAGE1_DATASET = 'kathbath'
LIBRISPEECH_HF = 'openslr/librispeech_asr'
LIBRISPEECH_PARQUET = 'clean/train.100/0000.parquet'
LIBRISPEECH_SR = 16000
KATHBATH_HF = 'ai4bharat/Kathbath'
KATHBATH_LANGUAGES = (
    'bengali', 'gujarati', 'hindi', 'kannada', 'malayalam', 'marathi',
    'odia', 'punjabi', 'sanskrit', 'tamil', 'telugu', 'urdu',
)
DEFAULT_KATHBATH_LANGUAGES = (
    'hindi', 'gujarati',
)


def _words_to_token_ids(text: str, max_words: int = 8) -> List[int]:
    """Unicode-aware word hashing (works for Indic scripts)."""
    words = [w for w in text.strip().lower().split() if w][:max_words]
    if not words:
        return [VOCAB.text_token(0)]
    return [VOCAB.text_token(abs(hash(w)) % 400) for w in words]


def decode_audio_field(audio: Any) -> Dict[str, Any]:
    """Decode LibriSpeech parquet audio (bytes or path) without torchcodec."""
    if isinstance(audio, dict) and 'array' in audio and audio['array'] is not None:
        return {
            'array': np.asarray(audio['array'], dtype=np.float32),
            'sampling_rate': int(audio.get('sampling_rate', LIBRISPEECH_SR)),
        }
    if isinstance(audio, dict):
        if audio.get('bytes'):
            arr, sr = sf.read(io.BytesIO(audio['bytes']))
            return {'array': arr.astype(np.float32), 'sampling_rate': int(sr)}
        if audio.get('path'):
            arr, sr = sf.read(audio['path'])
            return {'array': arr.astype(np.float32), 'sampling_rate': int(sr)}
    raise ValueError(f'Cannot decode audio field: {type(audio)} keys={getattr(audio, "keys", lambda: [])()}')


def _load_librispeech_rows(n_pairs: int, seed: int = 42) -> List[Dict]:
    """Load utterances from Parquet shard (avoids deprecated dataset loading script)."""
    from datasets import Audio, load_dataset
    from huggingface_hub import hf_hub_download

    need = max(n_pairs * 2 + 4, 16)
    parquet_path = hf_hub_download(
        repo_id=LIBRISPEECH_HF,
        filename=LIBRISPEECH_PARQUET,
        repo_type='dataset',
    )
    ds = load_dataset(
        'parquet',
        data_files={'train': parquet_path},
        split=f'train[:{need}]',
    )
    ds = ds.cast_column('audio', Audio(decode=False))

    rows = []
    for i in range(len(ds)):
        row = ds[i]
        rows.append({
            'text': row['text'],
            'audio': decode_audio_field(row['audio']),
        })

    rng = np.random.default_rng(seed)
    if len(rows) > need:
        idx = rng.choice(len(rows), size=need, replace=False)
        rows = [rows[int(i)] for i in idx]

    pairs = []
    for i in range(0, len(rows) - 1, 2):
        pairs.append({'user': rows[i], 'assistant': rows[i + 1]})
        if len(pairs) >= n_pairs:
            break
    if len(pairs) < max(1, n_pairs // 4):
        raise RuntimeError(
            f'LibriSpeech load yielded only {len(pairs)} pairs (need ~{n_pairs}). '
            f'Check network/cache for {LIBRISPEECH_HF}/{LIBRISPEECH_PARQUET}.'
        )
    return pairs


def _kathbath_parquet_shards(lang: str) -> List[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(KATHBATH_HF, repo_type='dataset')
    shards = sorted(
        f for f in files
        if f.startswith(f'{lang}/train-') and f.endswith('.parquet')
    )
    if not shards:
        raise RuntimeError(f'No Kathbath parquet shards for language {lang!r}')
    return shards


def _load_kathbath_rows(lang: str, n_pairs: int, seed: int = 42) -> List[Dict]:
    """Load utterance pairs from Kathbath parquet (bytes embedded; no torchcodec)."""
    from huggingface_hub import hf_hub_download

    need = max(n_pairs * 2 + 4, 16)
    rows: List[Dict] = []
    for shard in _kathbath_parquet_shards(lang):
        parquet_path = hf_hub_download(KATHBATH_HF, shard, repo_type='dataset')
        table = pq.read_table(parquet_path, columns=['text', 'audio_filepath'])
        for i in range(table.num_rows):
            audio = decode_audio_field(table['audio_filepath'][i].as_py())
            rows.append({
                'text': table['text'][i].as_py(),
                'audio': audio,
                'lang': lang,
            })
            if len(rows) >= need:
                break
        if len(rows) >= need:
            break

    if len(rows) < 4:
        raise RuntimeError(
            f'Kathbath {lang} yielded only {len(rows)} rows (need ~{need}). '
            f'Check cache for {KATHBATH_HF}.'
        )

    rng = np.random.default_rng(seed + abs(hash(lang)) % 10000)
    if len(rows) > need:
        idx = rng.choice(len(rows), size=need, replace=False)
        rows = [rows[int(i)] for i in idx]

    pairs = []
    for i in range(0, len(rows) - 1, 2):
        pairs.append({'user': rows[i], 'assistant': rows[i + 1], 'lang': lang})
        if len(pairs) >= n_pairs:
            break
    if len(pairs) < max(1, n_pairs // 4):
        raise RuntimeError(
            f'Kathbath {lang} yielded only {len(pairs)} pairs (need ~{n_pairs}).'
        )
    return pairs


def _build_duplex_sample(
    pair: Dict,
    rng: np.random.Generator,
    barge_in_prob: float,
    max_audio_chunks: int,
) -> Dict:
    user = pair['user']
    asst = pair['assistant']
    user_toks = _words_to_token_ids(user['text'])
    asst_toks = _words_to_token_ids(asst['text'])

    if rng.random() < barge_in_prob:
        kind = ScenarioKind.BARGE_IN
        block = build_block(kind, rng, truncate_reply=max(1, len(asst_toks) // 2))
        block.env_tokens = user_toks[:4]
    elif rng.random() < 0.5:
        kind = ScenarioKind.USER_TURN
        block = build_block(kind, rng)
        block.env_tokens = user_toks[:4]
    else:
        kind = ScenarioKind.ASSISTANT_TURN
        block = build_block(kind, rng)
        block.reply_tokens = asst_toks[:6]

    inp, lab = block_to_token_lists(block)
    audio_slot_indices = []
    for i, tok in enumerate(inp):
        if tok == VOCAB.env_mark:
            insert_at = i + 1
            n_slots = min(max_audio_chunks, max(1, len(user_toks)))
            placeholders = [VOCAB.pad] * n_slots
            inp = inp[:insert_at] + placeholders + inp[insert_at:]
            lab = lab[:insert_at] + [-100] * n_slots + lab[insert_at:]
            audio_slot_indices = list(range(insert_at, insert_at + n_slots))
            break

    sample = {
        'input_ids': inp,
        'labels': lab,
        'user_audio': user['audio'],
        'assistant_audio': asst['audio'],
        'kind': kind.name,
        'audio_slot_indices': audio_slot_indices,
    }
    if 'lang' in pair:
        sample['lang'] = pair['lang']
    return sample


def _pairs_to_samples(
    pairs: List[Dict],
    seed: int,
    barge_in_prob: float,
    max_audio_chunks: int,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    return [_build_duplex_sample(p, rng, barge_in_prob, max_audio_chunks) for p in pairs]


class LibriSpeechDuplexDataset(Dataset):
    """Simulated duplex from LibriSpeech pairs (user clip + assistant clip + transcripts)."""

    def __init__(
        self,
        n_pairs: int = 512,
        seed: int = 42,
        barge_in_prob: float = 0.3,
        max_audio_chunks: int = 4,
    ):
        self.max_audio_chunks = max_audio_chunks
        self.pairs = _load_librispeech_rows(n_pairs, seed=seed)
        self.samples = _pairs_to_samples(
            self.pairs, seed, barge_in_prob, max_audio_chunks,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class KathbathDuplexDataset(Dataset):
    """Multilingual duplex pairs from ai4bharat/Kathbath."""

    def __init__(
        self,
        languages: Sequence[str] = DEFAULT_KATHBATH_LANGUAGES,
        n_pairs_per_lang: int = 100,
        seed: int = 42,
        barge_in_prob: float = 0.3,
        max_audio_chunks: int = 4,
    ):
        self.max_audio_chunks = max_audio_chunks
        langs = [l.strip().lower() for l in languages if l.strip()]
        unknown = [l for l in langs if l not in KATHBATH_LANGUAGES]
        if unknown:
            raise ValueError(
                f'Unknown Kathbath languages: {unknown}. '
                f'Valid: {list(KATHBATH_LANGUAGES)}'
            )
        pairs: List[Dict] = []
        for lang in langs:
            print(f'  Loading Kathbath {lang} ({n_pairs_per_lang} pairs)...')
            pairs.extend(_load_kathbath_rows(lang, n_pairs_per_lang, seed=seed))
        self.pairs = pairs
        self.samples = _pairs_to_samples(
            self.pairs, seed, barge_in_prob, max_audio_chunks,
        )
        print(f'  Kathbath total: {len(self.samples)} samples ({len(langs)} langs)')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class MixedDuplexDataset(Dataset):
    """Concatenate multiple duplex datasets (e.g. LibriSpeech + Kathbath)."""

    def __init__(self, datasets: Sequence[Dataset]):
        self.samples: List[Dict] = []
        for ds in datasets:
            self.samples.extend(ds.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def parse_languages(languages: str) -> List[str]:
    if not languages.strip():
        return list(DEFAULT_KATHBATH_LANGUAGES)
    return [p.strip().lower() for p in languages.split(',') if p.strip()]


def build_stage1_dataset(
    dataset: str,
    n_pairs: int = 800,
    seed: int = 42,
    languages: Optional[str] = None,
    n_pairs_per_lang: Optional[int] = None,
    barge_in_prob: float = 0.3,
    max_audio_chunks: int = 4,
) -> Dataset:
    """Factory for Stage 1 duplex datasets."""
    name = dataset.lower()
    if name == 'librispeech':
        return LibriSpeechDuplexDataset(
            n_pairs=n_pairs, seed=seed,
            barge_in_prob=barge_in_prob, max_audio_chunks=max_audio_chunks,
        )
    if name == 'kathbath':
        langs = parse_languages(languages or '')
        npl = n_pairs_per_lang or max(50, n_pairs // max(1, len(langs)))
        return KathbathDuplexDataset(
            languages=langs, n_pairs_per_lang=npl, seed=seed,
            barge_in_prob=barge_in_prob, max_audio_chunks=max_audio_chunks,
        )
    if name == 'mix':
        langs = parse_languages(languages or '')
        ls_n = max(100, n_pairs // 2)
        kb_n = n_pairs - ls_n
        npl = n_pairs_per_lang or max(50, kb_n // max(1, len(langs)))
        return MixedDuplexDataset([
            LibriSpeechDuplexDataset(
                n_pairs=ls_n, seed=seed,
                barge_in_prob=barge_in_prob, max_audio_chunks=max_audio_chunks,
            ),
            KathbathDuplexDataset(
                languages=langs, n_pairs_per_lang=npl, seed=seed + 1,
                barge_in_prob=barge_in_prob, max_audio_chunks=max_audio_chunks,
            ),
        ])
    raise ValueError(
        f'Unknown dataset {dataset!r}. Use librispeech | kathbath | mix.'
    )


# ── Single-utterance ASR/TTS rows (Stage A/B; unified-vocab pipeline) ─────────

def _load_kathbath_utterances(lang: str, n: int, seed: int = 42) -> List[Dict]:
    """Load individual (audio, text, lang) utterances from Kathbath parquet."""
    from huggingface_hub import hf_hub_download

    need = max(n + 4, 8)
    rows: List[Dict] = []
    for shard in _kathbath_parquet_shards(lang):
        parquet_path = hf_hub_download(KATHBATH_HF, shard, repo_type='dataset')
        table = pq.read_table(parquet_path, columns=['text', 'audio_filepath'])
        for i in range(table.num_rows):
            text = table['text'][i].as_py()
            if not text:
                continue
            audio = decode_audio_field(table['audio_filepath'][i].as_py())
            rows.append({'text': text, 'audio': audio, 'lang': lang})
            if len(rows) >= need:
                break
        if len(rows) >= need:
            break
    if len(rows) < max(1, n // 4):
        raise RuntimeError(f'Kathbath {lang}: only {len(rows)} utterances (need ~{n}).')
    rng = np.random.default_rng(seed + abs(hash(lang)) % 10000)
    if len(rows) > n:
        idx = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[int(i)] for i in idx]
    return rows


def _load_librispeech_utterances(n: int, seed: int = 42) -> List[Dict]:
    """Load individual (audio, text, en) utterances from a LibriSpeech parquet shard."""
    from datasets import Audio, load_dataset
    from huggingface_hub import hf_hub_download

    need = max(n + 4, 8)
    parquet_path = hf_hub_download(
        repo_id=LIBRISPEECH_HF, filename=LIBRISPEECH_PARQUET, repo_type='dataset',
    )
    ds = load_dataset('parquet', data_files={'train': parquet_path},
                      split=f'train[:{need}]')
    ds = ds.cast_column('audio', Audio(decode=False))
    rows = []
    for i in range(len(ds)):
        row = ds[i]
        if not row['text']:
            continue
        rows.append({
            'text': row['text'].lower(),
            'audio': decode_audio_field(row['audio']),
            'lang': 'english',
        })
    rng = np.random.default_rng(seed)
    if len(rows) > n:
        idx = rng.choice(len(rows), size=n, replace=False)
        rows = [rows[int(i)] for i in idx]
    return rows


def load_asr_rows(
    languages: Sequence[str] = ('hindi', 'gujarati'),
    n_per_lang: int = 2000,
    include_english: bool = True,
    n_english: int = 2000,
    seed: int = 42,
) -> List[Dict]:
    """Collect single-utterance (audio, text, lang) rows for Stage A/B training."""
    rows: List[Dict] = []
    for lang in languages:
        lang = lang.strip().lower()
        if lang not in KATHBATH_LANGUAGES:
            raise ValueError(f'Unknown Kathbath language {lang!r}')
        print(f'  ASR rows: kathbath {lang} ({n_per_lang})...')
        rows.extend(_load_kathbath_utterances(lang, n_per_lang, seed=seed))
    if include_english and n_english > 0:
        print(f'  ASR rows: librispeech english ({n_english})...')
        rows.extend(_load_librispeech_utterances(n_english, seed=seed))
    print(f'  ASR rows total: {len(rows)}')
    return rows


def collate_stage1(batch: List[Dict], pad_id: int = VOCAB.pad) -> Dict[str, torch.Tensor]:
    max_len = max(len(s['input_ids']) for s in batch)
    max_slots = max(len(s['audio_slot_indices']) for s in batch)
    b = len(batch)
    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    audio_positions = torch.full((b, max_slots), -1, dtype=torch.long)
    for i, s in enumerate(batch):
        n = len(s['input_ids'])
        input_ids[i, :n] = torch.tensor(s['input_ids'], dtype=torch.long)
        labels[i, :n] = torch.tensor(s['labels'], dtype=torch.long)
        for j, pos in enumerate(s['audio_slot_indices']):
            audio_positions[i, j] = pos
    return {
        'input_ids': input_ids,
        'labels': labels,
        'audio_positions': audio_positions,
        'raw_batch': batch,
    }
