"""Stage D: full-duplex streaming inference for the unified voice model.

One PAM recurrent state spans the whole session (O(1)/token, no growing cache),
carrying acoustic/conversational gist across ~320 ms time-multiplexed blocks that
hold both incoming mic frames and outgoing codec tokens — so the model listens
*while* it speaks. Verbatim history lives in the LLM brain's text context (the
two-layer memory in the plan); the 100M state only holds gist.

Turn pipeline:
    mic audio ->(S2T, state-carry) transcript
             ->(brain LLM) reply text
             ->(T2S, state-carry) codec tokens -> Mimi decoder -> speaker

Barge-in: pass `interrupt_check` to `speak`/`converse`; when it returns True the
codec emission truncates and control flips to <listen> mid-reply (the scenario
Stage C trains).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from v11.duplex.codec import MimiCodec, MIMI_SR, delay_unflatten
from v11.duplex.encoder import FrozenWhisperEncoder
from v11.duplex.model import V11DuplexLM
from v11.duplex.tokenizer import (
    AUDIO_PAD, ENV_MARK, EOS, LANG_TOKEN, LISTEN, LLM, SPEAK, TRANSCRIBE, TTS,
    DuplexTokenizer,
)


BLOCK_MS = 320  # ~4 frames @ 12.5 Hz per time-multiplexed block


@dataclass
class LoadedSession:
    model: V11DuplexLM
    encoder: FrozenWhisperEncoder
    codec: MimiCodec
    tokenizer: DuplexTokenizer
    device: torch.device


def load_session(
    checkpoint: str,
    tokenizer_dir: str = '',
    whisper: str = 'openai/whisper-small',
    mimi: str = 'kyutai/mimi',
    device: Optional[str] = None,
) -> LoadedSession:
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    cfg = ckpt['config']
    tok_dir = tokenizer_dir or ckpt.get('tokenizer_dir', '')
    if not tok_dir:
        raise ValueError('tokenizer_dir not in checkpoint; pass --tokenizer_dir')
    tokenizer = DuplexTokenizer.load(tok_dir)
    encoder = FrozenWhisperEncoder(whisper, device=str(dev))
    codec = MimiCodec(tokenizer.vocab, model_name=mimi, device=str(dev))
    model = V11DuplexLM(cfg, audio_feat_dim=encoder.out_dim).to(dev)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return LoadedSession(model, encoder, codec, tokenizer, dev)


class DuplexSession:
    """Stateful streaming conversation: one PAM state + one brain context."""

    def __init__(self, loaded: LoadedSession, brain=None, system_prompt: Optional[str] = None):
        self.s = loaded
        self.device = loaded.device
        self.vocab = loaded.tokenizer.vocab
        self.states: Optional[List] = None
        self.step: int = 0
        self.brain = brain
        if brain is not None:
            from v11.duplex.brain import Conversation, DEFAULT_SYSTEM
            self.conv = Conversation(system_prompt or DEFAULT_SYSTEM)
        else:
            self.conv = None
        # codec-only logits mask (+eos) for constrained speech generation
        self._codec_mask = torch.full((self.vocab.total_size,), float('-inf'), device=self.device)
        self._codec_mask[self.vocab.codec_offset:self.vocab.total_size] = 0.0
        self._codec_mask[EOS] = 0.0

    def reset(self, keep_brain: bool = False):
        self.states, self.step = None, 0
        if self.conv is not None and not keep_brain:
            self.conv.turns.clear()

    # ── low-level state-carrying feed / decode ─────────────────────────────
    @torch.no_grad()
    def _feed(self, ids: List[int], audio_embeds=None, audio_positions=None) -> torch.Tensor:
        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, self.states, _ = self.s.model(
            ids_t, states=self.states, step_offset=self.step,
            audio_embeds=audio_embeds, audio_positions=audio_positions)
        self.step += ids_t.shape[1]
        return logits

    @torch.no_grad()
    def _decode(self, first_logits: torch.Tensor, max_new: int,
                allow_mask: Optional[torch.Tensor] = None,
                stop_ids=(EOS,), interrupt_check: Optional[Callable[[], bool]] = None
                ) -> List[int]:
        out: List[int] = []
        logits = first_logits[:, -1]
        if allow_mask is not None:
            logits = logits + allow_mask
        nxt = int(logits.argmax(-1))
        for _ in range(max_new):
            if nxt in stop_ids:
                break
            if interrupt_check is not None and interrupt_check():
                break
            out.append(nxt)
            step_logits = self._feed([nxt])[:, -1]
            if allow_mask is not None:
                step_logits = step_logits + allow_mask
            nxt = int(step_logits.argmax(-1))
        return out

    # ── S2T: mic audio -> transcript (block-streamed, state carried) ───────
    @torch.no_grad()
    def transcribe(self, waveform: np.ndarray, sample_rate: int, lang: str = 'en',
                   block_ms: int = BLOCK_MS, max_new: int = 200) -> str:
        lang_id = LANG_TOKEN.get(str(lang).lower(), LANG_TOKEN['en'])
        frames = self.s.encoder.encode_frames(
            torch.tensor(np.asarray(waveform), dtype=torch.float32), sample_rate)
        frames = frames.to(self.device)

        self._feed([LISTEN, ENV_MARK])
        fps = self.s.encoder.ENCODER_FPS / 4.0  # stride-4 -> ~12.5 Hz
        block = max(1, int(round(fps * block_ms / 1000.0)))
        for i in range(0, frames.shape[0], block):  # time-multiplexed blocks
            chunk = frames[i:i + block]
            n = chunk.shape[0]
            embeds = self.s.model.project_audio(chunk.unsqueeze(0))
            positions = torch.arange(n, device=self.device).unsqueeze(0)
            self._feed([AUDIO_PAD] * n, audio_embeds=embeds, audio_positions=positions)

        logits = self._feed([TRANSCRIBE, lang_id])
        toks = self._decode(logits, max_new=max_new)
        return self.s.tokenizer.decode_text(toks)

    # ── LLM brain ──────────────────────────────────────────────────────────
    def think(self, user_text: str, stream: bool = False):
        if self.brain is None:
            return user_text  # echo fallback when no brain configured
        self.conv.add_user(user_text)
        if stream:
            return self.brain.chat(self.conv.messages(), stream=True)
        reply = self.brain.chat(self.conv.messages(), stream=False)
        self.conv.add_assistant(reply)
        return reply

    # ── T2S: reply text -> codec -> waveform (state carried, barge-in) ─────
    @torch.no_grad()
    def speak(self, reply_text: str, lang: str = 'en', max_codec_tokens: int = 500,
              interrupt_check: Optional[Callable[[], bool]] = None
              ) -> Tuple[np.ndarray, int]:
        lang_id = LANG_TOKEN.get(str(lang).lower(), LANG_TOKEN['en'])
        text_ids = self.s.tokenizer.encode_text(reply_text)
        logits = self._feed([SPEAK, LLM, lang_id] + text_ids + [TTS])
        stream = self._decode(logits, max_new=max_codec_tokens, allow_mask=self._codec_mask,
                              interrupt_check=interrupt_check)
        if not stream:
            return np.zeros(0, dtype=np.float32), MIMI_SR
        codes = delay_unflatten(stream, self.vocab)
        if codes.shape[1] == 0:
            return np.zeros(0, dtype=np.float32), MIMI_SR
        wav = self.s.codec.decode(codes)
        return np.asarray(wav, dtype=np.float32), MIMI_SR

    # ── full turn ──────────────────────────────────────────────────────────
    def converse(self, waveform: np.ndarray, sample_rate: int, lang: str = 'en',
                 interrupt_check: Optional[Callable[[], bool]] = None) -> Dict:
        transcript = self.transcribe(waveform, sample_rate, lang=lang)
        reply = self.think(transcript)
        audio, sr = self.speak(reply, lang=lang, interrupt_check=interrupt_check)
        return {'transcript': transcript, 'reply': reply, 'audio': audio, 'sample_rate': sr}


def _cli():
    p = argparse.ArgumentParser(description='V11 duplex streaming inference')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer_dir', default='')
    p.add_argument('--audio', required=True, help='wav/flac file to converse with')
    p.add_argument('--lang', default='en')
    p.add_argument('--whisper', default='openai/whisper-small')
    p.add_argument('--mimi', default='kyutai/mimi')
    p.add_argument('--out', default='duplex_reply.wav')
    p.add_argument('--brain', action='store_true', help='use external LLM brain')
    args = p.parse_args()

    import soundfile as sf
    loaded = load_session(args.checkpoint, args.tokenizer_dir, args.whisper, args.mimi)
    brain = None
    if args.brain:
        from v11.duplex.brain import BrainClient
        brain = BrainClient()
    sess = DuplexSession(loaded, brain=brain)
    wav, sr = sf.read(args.audio)
    result = sess.converse(np.asarray(wav, dtype=np.float32), int(sr), lang=args.lang)
    print(f'transcript: {result["transcript"]!r}')
    print(f'reply     : {result["reply"]!r}')
    if result['audio'].size:
        sf.write(args.out, result['audio'], result['sample_rate'])
        print(f'spoken reply -> {args.out} ({result["audio"].size / result["sample_rate"]:.1f}s)')
    else:
        print('no speech generated (model likely undertrained)')


if __name__ == '__main__':
    _cli()
