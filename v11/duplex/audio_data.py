"""Stage 1 audio dataset (open fallback; Fisher LDC when available)."""

import io
import re
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from v11.duplex.interleave import ScenarioKind, block_to_token_lists, build_block
from v11.duplex.thinking import VOCAB

# Stage 1 data sources (see plan):
#   Primary (needs LDC license): Fisher English LDC97S62 dual-channel ~2000h
#   Fallback (open): LibriSpeech clean train.100 via Parquet (no HF loading script)
DEFAULT_STAGE1_DATASET = 'librispeech'
LIBRISPEECH_HF = 'openslr/librispeech_asr'
LIBRISPEECH_PARQUET = 'clean/train.100/0000.parquet'
LIBRISPEECH_SR = 16000


def _words_to_token_ids(text: str, max_words: int = 8) -> List[int]:
    words = re.findall(r"[a-zA-Z']+", text.lower())[:max_words]
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
        self.rng = np.random.default_rng(seed)
        self.pairs = _load_librispeech_rows(n_pairs, seed=seed)
        self.barge_in_prob = barge_in_prob
        self.samples: List[Dict] = []
        for pair in self.pairs:
            self.samples.append(self._build_sample(pair))

    def _build_sample(self, pair: Dict) -> Dict:
        user = pair['user']
        asst = pair['assistant']
        user_toks = _words_to_token_ids(user['text'])
        asst_toks = _words_to_token_ids(asst['text'])

        if self.rng.random() < self.barge_in_prob:
            kind = ScenarioKind.BARGE_IN
            block = build_block(kind, self.rng, truncate_reply=max(1, len(asst_toks) // 2))
            block.env_tokens = user_toks[:4]
        elif self.rng.random() < 0.5:
            kind = ScenarioKind.USER_TURN
            block = build_block(kind, self.rng)
            block.env_tokens = user_toks[:4]
        else:
            kind = ScenarioKind.ASSISTANT_TURN
            block = build_block(kind, self.rng)
            block.reply_tokens = asst_toks[:6]

        inp, lab = block_to_token_lists(block)
        audio_slot_indices = []
        for i, tok in enumerate(inp):
            if tok == VOCAB.env_mark:
                # Placeholder slots after env_mark for audio injection
                insert_at = i + 1
                n_slots = min(self.max_audio_chunks, max(1, len(user_toks)))
                placeholders = [VOCAB.pad] * n_slots
                inp = inp[:insert_at] + placeholders + inp[insert_at:]
                lab = lab[:insert_at] + [-100] * n_slots + lab[insert_at:]
                audio_slot_indices = list(range(insert_at, insert_at + n_slots))
                break

        return {
            'input_ids': inp,
            'labels': lab,
            'user_audio': user['audio'],
            'assistant_audio': asst['audio'],
            'kind': kind.name,
            'audio_slot_indices': audio_slot_indices,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


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
