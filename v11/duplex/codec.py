"""Mimi neural codec wrapper + acoustic delay-pattern (de)interleaving.

Speech OUT for the duplex model is generated as discrete Mimi codec tokens
(kyutai/mimi: 24 kHz, 12.5 Hz frame rate, 2048-way codebooks). The PAM backbone
emits codec tokens in the unified id-space; the frozen Mimi decoder renders the
waveform. We use the first `n_codebooks` quantizers (semantic + a few acoustic),
matching `DuplexVocab.n_codebooks`.

Delay pattern (Moshi / MusicGen style): codebook k is delayed by k frames so that
when the LM emits the K tokens of a frame left-to-right, higher codebooks are
predicted after (and can condition on) the lower ones. `delay_flatten` turns
codes [K, T] into a flat global-id stream; `delay_unflatten` inverts it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from v11.duplex.encoder import resample_audio, normalize_waveform
from v11.duplex.tokenizer import AUDIO_PAD, DuplexVocab

MIMI_HF = 'kyutai/mimi'
MIMI_SR = 24000


class MimiCodec:
    """Frozen Mimi encode/decode bound to the unified codec id-space."""

    def __init__(self, vocab: DuplexVocab, model_name: str = MIMI_HF,
                 device: Optional[str] = None):
        from transformers import MimiModel
        self.vocab = vocab
        self.model_name = model_name
        self.model = MimiModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.n_codebooks = vocab.n_codebooks
        if vocab.codebook_size != self.model.config.codebook_size:
            raise ValueError(
                f'vocab.codebook_size={vocab.codebook_size} != Mimi '
                f'{self.model.config.codebook_size}; rebuild the tokenizer layout.'
            )
        self.device = device
        if device is not None:
            self.model.to(device)

    # ── waveform <-> codes ─────────────────────────────────────────────────
    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """waveform [samples] -> codes LongTensor [n_codebooks, T] (cpu)."""
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        wf = normalize_waveform(waveform.detach().cpu().numpy())
        if int(sample_rate) != MIMI_SR:
            wf = resample_audio(wf, int(sample_rate), MIMI_SR)
        x = torch.tensor(wf, dtype=torch.float32).view(1, 1, -1)
        dev = next(self.model.parameters()).device
        out = self.model.encode(x.to(dev), num_quantizers=self.n_codebooks)
        codes = out.audio_codes if hasattr(out, 'audio_codes') else out[0]
        return codes[0, : self.n_codebooks].detach().cpu().long()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> np.ndarray:
        """codes [n_codebooks, T] -> waveform np.float32 at 24 kHz."""
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        dev = next(self.model.parameters()).device
        out = self.model.decode(codes.to(dev).long())
        audio = out.audio_values if hasattr(out, 'audio_values') else out[0]
        return audio.squeeze().detach().cpu().float().numpy()


# ── Delay pattern (pure id-space; no model needed) ───────────────────────────

def delay_flatten(codes: torch.Tensor, vocab: DuplexVocab,
                  empty_id: int = AUDIO_PAD) -> List[int]:
    """codes [K, T] -> flat global-id stream of length K*(T+K-1) (delay pattern)."""
    if codes.dim() != 2:
        raise ValueError('codes must be [K, T]')
    K, T = codes.shape
    if K != vocab.n_codebooks:
        raise ValueError(f'codes has {K} codebooks, vocab expects {vocab.n_codebooks}')
    S = T + (K - 1)
    stream: List[int] = []
    for s in range(S):
        for k in range(K):
            t = s - k
            if 0 <= t < T:
                stream.append(vocab.codec_to_global(k, int(codes[k, t])))
            else:
                stream.append(empty_id)
    return stream


def delay_unflatten(stream: Sequence[int], vocab: DuplexVocab,
                    empty_id: int = AUDIO_PAD) -> torch.Tensor:
    """Inverse of `delay_flatten`. Returns codes LongTensor [K, T].

    Robust to generation noise: tokens that don't decode to the expected
    codebook slot are treated as `empty` (code 0 fallback).
    """
    K = vocab.n_codebooks
    n = len(stream)
    S = n // K
    T = max(0, S - (K - 1))
    codes = torch.zeros(K, T, dtype=torch.long)
    for s in range(S):
        for k in range(K):
            gid = int(stream[s * K + k])
            t = s - k
            if not (0 <= t < T):
                continue
            if vocab.is_codec(gid):
                cb, code = vocab.global_to_codec(gid)
                codes[k, t] = code if cb == k else 0
    return codes


def flat_interleave(codes: torch.Tensor, vocab: DuplexVocab) -> List[int]:
    """Simple K-per-frame interleave (no delay); length K*T. Lossless fallback."""
    K, T = codes.shape
    return [vocab.codec_to_global(k, int(codes[k, t])) for t in range(T) for k in range(K)]


def flat_deinterleave(stream: Sequence[int], vocab: DuplexVocab) -> torch.Tensor:
    K = vocab.n_codebooks
    T = len(stream) // K
    codes = torch.zeros(K, T, dtype=torch.long)
    for t in range(T):
        for k in range(K):
            gid = int(stream[t * K + k])
            if vocab.is_codec(gid):
                cb, code = vocab.global_to_codec(gid)
                codes[k, t] = code if cb == k else 0
    return codes


def _selftest():
    """Round-trip the delay + flat patterns on random codes (no model download)."""
    vocab = DuplexVocab(n_text=32000, n_codebooks=4, codebook_size=2048)
    K, T = vocab.n_codebooks, 37
    codes = torch.randint(0, vocab.codebook_size, (K, T))

    stream = delay_flatten(codes, vocab)
    assert len(stream) == K * (T + K - 1), len(stream)
    back = delay_unflatten(stream, vocab)
    assert back.shape == (K, T), back.shape
    assert torch.equal(back, codes), 'delay round-trip mismatch'

    fs = flat_interleave(codes, vocab)
    assert len(fs) == K * T
    fb = flat_deinterleave(fs, vocab)
    assert torch.equal(fb, codes), 'flat round-trip mismatch'

    # all stream ids are valid codec ids or the empty marker
    assert all(vocab.is_codec(g) or g == AUDIO_PAD for g in stream)
    print('codec delay/flat round-trip: OK '
          f'(K={K} T={T} delay_len={len(stream)} flat_len={len(fs)})')


if __name__ == '__main__':
    _selftest()
