"""Unified vocabulary + SentencePiece tokenizer for the duplex voice model.

One flat id-space shared by the PAM backbone's tied embedding / LM head:

    [ control ]  [ text (SentencePiece) ]  [ codec (Mimi codebooks) ]
      0..15        16..16+n_text-1           after text .. +n_cb*cb_size

Rationale (see plan): GPT-2 BPE shatters Devanagari/Gujarati into bytes and a
50k vocab wastes ~38M of a 100M model in tied embeddings. A ~32k SentencePiece
BPE over hi+gu+en plus a compact codec block keeps the embedding ~31M while
covering Indic scripts natively (byte_fallback guarantees total coverage).

This module owns ONLY the id-space + text (de)tokenization. Mimi encode/decode
and the acoustic delay-pattern live in `v11/duplex/codec.py`; that module maps
(codebook, code) pairs through `DuplexVocab.codec_to_global` defined here.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ── Control tokens (fixed low block; never collide with text/codec) ──────────

PAD = 0
BOS = 1
EOS = 2
LISTEN = 3            # duplex control: keep listening
SPEAK = 4            # duplex control: start speaking
BACKCHANNEL = 5      # duplex control: short acknowledgement while user speaks
ENV_MARK = 6         # environment (user) audio stream marker; frames follow
AST_MARK = 7         # assistant audio echo marker
BLOCK_SEP = 8        # separator between time blocks
TRANSCRIBE = 9       # <transcribe>: begin S2T transcript text
LLM = 10             # <llm>: begin external-brain reply text
TTS = 11             # <tts>: begin codec token stream (spoken reply)
AUDIO_PAD = 12       # placeholder id at positions where audio embeds are injected
LANG_HI = 13
LANG_GU = 14
LANG_EN = 15

N_CONTROL = 16

LANG_TOKEN = {'hindi': LANG_HI, 'hi': LANG_HI,
              'gujarati': LANG_GU, 'gu': LANG_GU,
              'english': LANG_EN, 'en': LANG_EN}

CONTROL_NAMES = {
    PAD: '<pad>', BOS: '<bos>', EOS: '<eos>', LISTEN: '<listen>',
    SPEAK: '<speak>', BACKCHANNEL: '<backchannel>', ENV_MARK: '<env>',
    AST_MARK: '<ast>', BLOCK_SEP: '<sep>', TRANSCRIBE: '<transcribe>',
    LLM: '<llm>', TTS: '<tts>', AUDIO_PAD: '<audio_pad>',
    LANG_HI: '<hi>', LANG_GU: '<gu>', LANG_EN: '<en>',
}

THINKING_IDS: Tuple[int, ...] = (LISTEN, SPEAK, BACKCHANNEL)


# ── Unified vocab layout ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class DuplexVocab:
    """Offsets that partition the flat id-space into control / text / codec."""

    n_text: int = 32000
    n_codebooks: int = 4
    codebook_size: int = 2048
    n_control: int = N_CONTROL

    @property
    def text_offset(self) -> int:
        return self.n_control

    @property
    def codec_offset(self) -> int:
        return self.n_control + self.n_text

    @property
    def total_size(self) -> int:
        return self.n_control + self.n_text + self.n_codebooks * self.codebook_size

    # id-space helpers -------------------------------------------------------
    def text_to_global(self, piece_id: int) -> int:
        return self.text_offset + int(piece_id)

    def global_to_text(self, gid: int) -> int:
        return int(gid) - self.text_offset

    def codec_to_global(self, codebook: int, code: int) -> int:
        if not 0 <= codebook < self.n_codebooks:
            raise ValueError(f'codebook {codebook} out of range [0,{self.n_codebooks})')
        return self.codec_offset + int(codebook) * self.codebook_size + int(code)

    def global_to_codec(self, gid: int) -> Tuple[int, int]:
        off = int(gid) - self.codec_offset
        if off < 0:
            raise ValueError(f'gid {gid} is not a codec token')
        return off // self.codebook_size, off % self.codebook_size

    def is_control(self, gid: int) -> bool:
        return 0 <= int(gid) < self.n_control

    def is_text(self, gid: int) -> bool:
        return self.text_offset <= int(gid) < self.codec_offset

    def is_codec(self, gid: int) -> bool:
        return self.codec_offset <= int(gid) < self.total_size

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'DuplexVocab':
        keep = {k: d[k] for k in ('n_text', 'n_codebooks', 'codebook_size', 'n_control') if k in d}
        return cls(**keep)


# ── SentencePiece text tokenizer bound to the unified id-space ───────────────

_SP_MODEL_NAME = 'duplex_spm.model'
_LAYOUT_NAME = 'duplex_vocab.json'


class DuplexTokenizer:
    """SentencePiece text tokenizer that emits/consumes GLOBAL unified ids."""

    def __init__(self, sp_model_path: str, vocab: DuplexVocab):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.vocab = vocab
        if self.sp.get_piece_size() > vocab.n_text:
            raise ValueError(
                f'SentencePiece has {self.sp.get_piece_size()} pieces but layout '
                f'reserves only {vocab.n_text}; retrain or widen n_text.'
            )

    # text <-> global ids ----------------------------------------------------
    def encode_text(self, text: str, add_bos: bool = False,
                    add_eos: bool = False, lang: Optional[str] = None) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(BOS)
        if lang is not None:
            ids.append(LANG_TOKEN.get(lang.lower(), LANG_EN))
        ids.extend(self.vocab.text_to_global(p) for p in self.sp.encode(text, out_type=int))
        if add_eos:
            ids.append(EOS)
        return ids

    def decode_text(self, gids: Sequence[int]) -> str:
        pieces = [self.vocab.global_to_text(g) for g in gids if self.vocab.is_text(g)]
        if not pieces:
            return ''
        return self.sp.decode(pieces)

    def pretty(self, gids: Sequence[int]) -> str:
        """Human-readable render mixing control names + decoded text spans."""
        out: List[str] = []
        run: List[int] = []
        for g in gids:
            if self.vocab.is_text(g):
                run.append(g)
                continue
            if run:
                out.append(self.decode_text(run))
                run = []
            if self.vocab.is_control(g):
                out.append(CONTROL_NAMES.get(int(g), f'<ctl{g}>'))
            elif self.vocab.is_codec(g):
                cb, code = self.vocab.global_to_codec(g)
                out.append(f'<c{cb}:{code}>')
        if run:
            out.append(self.decode_text(run))
        return ' '.join(s for s in out if s)

    # persistence ------------------------------------------------------------
    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        import shutil
        src = self.sp.serialized_model_proto()
        with open(out / _SP_MODEL_NAME, 'wb') as f:
            f.write(src)
        with open(out / _LAYOUT_NAME, 'w') as f:
            json.dump(self.vocab.to_dict(), f, indent=2)

    @classmethod
    def load(cls, out_dir: str) -> 'DuplexTokenizer':
        out = Path(out_dir)
        with open(out / _LAYOUT_NAME) as f:
            vocab = DuplexVocab.from_dict(json.load(f))
        return cls(str(out / _SP_MODEL_NAME), vocab)


# ── Training ─────────────────────────────────────────────────────────────────

def train_tokenizer(
    corpus_path: str,
    out_dir: str,
    n_text: int = 32000,
    n_codebooks: int = 4,
    codebook_size: int = 2048,
    model_type: str = 'unigram',
    character_coverage: float = 1.0,
    seed_sentencepiece_size: int = 1000000,
) -> DuplexTokenizer:
    """Train a SentencePiece model on `corpus_path` (one sentence per line).

    Uses byte_fallback so any hi/gu/en (or unseen) character is representable.
    Produces `duplex_spm.model` + `duplex_vocab.json` in `out_dir`.
    """
    import sentencepiece as spm

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_prefix = str(out / 'duplex_spm')
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=n_text,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=True,
        unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,
        input_sentence_size=seed_sentencepiece_size,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count() or 4,
    )
    vocab = DuplexVocab(n_text=n_text, n_codebooks=n_codebooks, codebook_size=codebook_size)
    tok = DuplexTokenizer(model_prefix + '.model', vocab)
    tok.save(out_dir)
    return tok


def write_corpus(lines: Iterable[str], corpus_path: str) -> int:
    """Write cleaned lines (one sentence per line) to a training corpus file."""
    Path(corpus_path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for line in lines:
            s = ' '.join(str(line).split())
            if s:
                f.write(s + '\n')
                n += 1
    return n


def collect_asr_transcripts(
    languages: Sequence[str] = ('hindi', 'gujarati'),
    n_per_lang: int = 20000,
    include_english: bool = True,
    n_english: int = 20000,
    seed: int = 42,
) -> List[str]:
    """Pull transcript text from Kathbath (hi/gu) + LibriSpeech (en) for tokenizer training.

    Text-only (audio decode skipped) so this is cheap relative to Stage A/B.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    from v11.duplex.audio_data import (
        KATHBATH_HF, _kathbath_parquet_shards, _load_librispeech_transcripts,
    )
    from v11.duplex.logutil import log

    lines: List[str] = []
    for lang in languages:
        got = 0
        for shard in _kathbath_parquet_shards(lang):
            path = hf_hub_download(KATHBATH_HF, shard, repo_type='dataset')
            table = pq.read_table(path, columns=['text'])
            for i in range(table.num_rows):
                t = table['text'][i].as_py()
                if t:
                    lines.append(t)
                    got += 1
                if got >= n_per_lang:
                    break
            if got >= n_per_lang:
                break
        log(f'tokenizer corpus: {lang} -> {got} lines')

    if include_english:
        en_lines = _load_librispeech_transcripts(n_english, seed=seed)
        lines.extend(en_lines)
        log(f'tokenizer corpus: english -> {len(en_lines)} lines')

    return lines


def _cli():
    import argparse
    p = argparse.ArgumentParser(description='Train the duplex unified tokenizer')
    p.add_argument('--out_dir', default='checkpoints_v11_duplex_tokenizer')
    p.add_argument('--corpus', default='', help='existing corpus file (one sentence/line)')
    p.add_argument('--languages', default='hindi,gujarati')
    p.add_argument('--n_per_lang', type=int, default=20000)
    p.add_argument('--n_english', type=int, default=20000)
    p.add_argument('--n_text', type=int, default=32000)
    p.add_argument('--n_codebooks', type=int, default=4)
    p.add_argument('--codebook_size', type=int, default=2048)
    p.add_argument('--model_type', default='unigram', choices=['unigram', 'bpe'])
    args = p.parse_args()

    from v11.duplex.logutil import elapsed_since, log
    import time

    corpus = args.corpus
    if not corpus:
        corpus = str(Path(args.out_dir) / 'corpus.txt')
        langs = [s.strip() for s in args.languages.split(',') if s.strip()]
        t0 = time.time()
        lines = collect_asr_transcripts(
            languages=langs, n_per_lang=args.n_per_lang,
            n_english=args.n_english,
        )
        n = write_corpus(lines, corpus)
        log(f'Wrote {n} lines -> {corpus} in {elapsed_since(t0)}')

    t1 = time.time()
    tok = train_tokenizer(
        corpus, args.out_dir, n_text=args.n_text,
        n_codebooks=args.n_codebooks, codebook_size=args.codebook_size,
        model_type=args.model_type,
    )
    log(f'Tokenizer saved to {args.out_dir}: total vocab = {tok.vocab.total_size} '
        f'(train {elapsed_since(t1)})')
    demo = 'नमस्ते, आप कैसे हैं? hello there. કેમ છો?'
    ids = tok.encode_text(demo, lang='hindi')
    log(f'demo roundtrip: {tok.decode_text(ids)}')


if __name__ == '__main__':
    _cli()
