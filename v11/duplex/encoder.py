"""Frozen Whisper encoder for Stage 1 (codec-free audio embeddings)."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import WhisperFeatureExtractor, WhisperModel


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
        wf = waveform.detach().cpu().numpy()
        sr = sample_rate
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
