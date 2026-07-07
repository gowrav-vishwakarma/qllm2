"""Frozen Whisper encoder for Stage 1 (codec-free audio embeddings)."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import WhisperFeatureExtractor, WhisperModel

WHISPER_SR = 16000


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int = WHISPER_SR) -> np.ndarray:
    """Linear resample to Whisper's expected sample rate."""
    wf = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr:
        return wf
    if wf.size == 0:
        return wf
    duration = wf.size / orig_sr
    n_out = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, 1.0, num=wf.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, wf).astype(np.float32)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Gradio mic/upload may be float or int16-scale; ensure float32 ~[-1, 1]."""
    wf = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wf.size == 0:
        return wf
    peak = float(np.max(np.abs(wf)))
    if peak > 1.5:
        wf = wf / 32768.0
    elif peak > 0:
        wf = wf / peak
    return wf.astype(np.float32)


class FrozenWhisperEncoder(nn.Module):
    """Whisper encoder only; weights frozen. Outputs mean-pooled frame vectors."""

    def __init__(self, model_name: str = 'openai/whisper-small', device: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.out_dim = whisper.config.d_model
        if device is not None:
            self.to(device)

    # Whisper: 16 kHz -> 100 fps mel -> conv stride 2 -> ~50 fps encoder frames.
    ENCODER_FPS = 50

    @torch.no_grad()
    def encode_frames(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        stride: int = 4,
        max_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """waveform [samples] -> [T', out_dim] frame sequence at ~ENCODER_FPS/stride.

        Unlike `encode_waveform` (1 mean-pooled vec/sec, throws away phonetics),
        this keeps the encoder's frame sequence, stride-pooled to ~12.5 Hz
        (stride=4). This is the input mode used for real ASR / duplex training.

        Whisper's feature extractor zero-pads every clip to 30 s (1500 frames);
        we crop to the frames that correspond to the true audio duration before
        pooling so short utterances don't carry ~1500 padded frames.
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        wf = normalize_waveform(waveform.detach().cpu().numpy())
        sr = int(sample_rate)
        if sr != WHISPER_SR:
            wf = resample_audio(wf, sr, WHISPER_SR)
            sr = WHISPER_SR
        if wf.size == 0:
            dev = next(self.encoder.parameters()).device
            return torch.zeros(1, self.out_dim, device=dev)

        duration = wf.size / sr
        valid_frames = max(1, int(round(duration * self.ENCODER_FPS)))

        inputs = self.feature_extractor(wf, sampling_rate=sr, return_tensors='pt')
        input_features = inputs.input_features.to(next(self.encoder.parameters()).device)
        hidden = self.encoder(input_features).last_hidden_state  # [1, 1500, D]
        frames = hidden.squeeze(0)
        frames = frames[:min(valid_frames, frames.shape[0])]  # crop pad frames

        if stride > 1:
            n = (frames.shape[0] // stride) * stride
            if n == 0:  # very short clip -> single pooled vector
                pooled = frames.mean(dim=0, keepdim=True)
            else:
                pooled = frames[:n].reshape(n // stride, stride, self.out_dim).mean(dim=1)
                if frames.shape[0] > n:  # tail remainder -> one more pooled frame
                    tail = frames[n:].mean(dim=0, keepdim=True)
                    pooled = torch.cat([pooled, tail], dim=0)
        else:
            pooled = frames

        if max_frames is not None and pooled.shape[0] > max_frames:
            pooled = pooled[:max_frames]
        return pooled

    @torch.no_grad()
    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_sec: float = 1.0,
        max_chunks: int = 4,
    ) -> torch.Tensor:
        """waveform [samples] float32 -> [n_chunks, out_dim] on encoder device."""
        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        wf = normalize_waveform(waveform.detach().cpu().numpy())
        sr = int(sample_rate)
        if sr != WHISPER_SR:
            wf = resample_audio(wf, sr, WHISPER_SR)
            sr = WHISPER_SR
        chunk_samples = max(1, int(chunk_sec * sr))
        chunks = []
        for start in range(0, len(wf), chunk_samples):
            piece = wf[start:start + chunk_samples]
            if len(piece) < chunk_samples // 4:
                continue
            inputs = self.feature_extractor(
                piece, sampling_rate=sr, return_tensors='pt',
            )
            input_features = inputs.input_features.to(next(self.encoder.parameters()).device)
            hidden = self.encoder(input_features).last_hidden_state  # [1, T, D]
            chunks.append(hidden.mean(dim=1).squeeze(0))
            if len(chunks) >= max_chunks:
                break
        if not chunks:
            dev = next(self.encoder.parameters()).device
            return torch.zeros(1, self.out_dim, device=dev)
        return torch.stack(chunks, dim=0)
