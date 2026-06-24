"""Streaming duplex inference: audio (or text proxy) -> thinking-token prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from v11.duplex.audio_data import _words_to_token_ids
from v11.duplex.config import get_duplex_config
from v11.duplex.encoder import FrozenWhisperEncoder
from v11.duplex.model import V11DuplexLM
from v11.duplex.thinking import VOCAB

THINKING_NAMES = {
    VOCAB.listen: 'listen',
    VOCAB.speak: 'speak',
    VOCAB.backchannel: 'backchannel',
}


@dataclass
class DuplexPrediction:
    thinking: str
    thinking_id: int
    probs: Dict[str, float]
    n_audio_chunks: int
    has_audio: bool
    checkpoint: str
    preset: str

    def to_markdown(self) -> str:
        lines = [
            f'**Prediction:** `{self.thinking}`',
            '',
            '| Token | Probability |',
            '|-------|-------------|',
        ]
        for name in ('listen', 'speak', 'backchannel'):
            p = self.probs.get(name, 0.0)
            bar = '█' * int(p * 20)
            lines.append(f'| `{name}` | {p:.1%} {bar} |')
        lines.extend([
            '',
            f'Audio chunks: {self.n_audio_chunks} | '
            f'Stage: {"1 (Whisper)" if self.has_audio else "0 (text proxy)"}',
            f'Checkpoint: `{self.checkpoint}`',
        ])
        return '\n'.join(lines)


def _has_audio_weights(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith('audio_proj.') for k in state_dict)


def discover_checkpoints(root: Union[str, Path] = '.') -> List[str]:
    """Find duplex checkpoint files under checkpoints_v11_duplex*."""
    root = Path(root)
    found: List[str] = []
    for d in sorted(root.glob('checkpoints_v11_duplex*')):
        if not d.is_dir():
            continue
        for name in ('best_model.pt', 'latest.pt'):
            p = d / name
            if p.is_file():
                found.append(str(p.resolve()))
    return found


def checkpoint_label(path: str) -> str:
    p = Path(path)
    return f'{p.parent.name}/{p.name}'


def load_checkpoint_meta(checkpoint: str) -> Dict:
    ckpt_dir = Path(checkpoint).parent
    metrics_path = ckpt_dir / 'metrics.json'
    if metrics_path.is_file():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def load_duplex_from_checkpoint(
    checkpoint: str,
    preset: Optional[str] = None,
    whisper: str = 'openai/whisper-small',
    device: Optional[str] = None,
) -> Tuple[V11DuplexLM, Optional[FrozenWhisperEncoder], bool, str]:
    """Load model (+ optional Whisper encoder) from a .pt checkpoint."""
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))

    meta = load_checkpoint_meta(checkpoint)
    cfg = ckpt.get('config')
    if cfg is not None and hasattr(cfg, 'dim'):
        preset_name = preset or meta.get('preset', 'duplex_5m')
    else:
        preset_name = preset or meta.get('preset', 'duplex_5m')
        cfg = get_duplex_config(preset_name)

    has_audio = _has_audio_weights(state)
    encoder = None
    audio_dim = 0
    if has_audio:
        encoder = FrozenWhisperEncoder(whisper, device=str(dev))
        audio_dim = encoder.out_dim

    model = V11DuplexLM(cfg, audio_feat_dim=audio_dim).to(dev)
    model.load_state_dict(state, strict=False)
    model.eval()
    if encoder is not None:
        encoder.eval()

    return model, encoder, has_audio, preset_name


@torch.no_grad()
def predict_thinking(
    model: V11DuplexLM,
    encoder: Optional[FrozenWhisperEncoder],
    audio: Optional[Tuple[int, np.ndarray]] = None,
    text_context: str = '',
    max_audio_chunks: int = 4,
    device: Optional[torch.device] = None,
    checkpoint: str = '',
    preset: str = 'duplex_5m',
) -> DuplexPrediction:
    """Predict listen / speak / backchannel from user audio or text proxy."""
    dev = device or next(model.parameters()).device
    has_audio = encoder is not None and model.audio_proj is not None
    n_chunks = 0
    audio_embeds = None
    audio_positions = None

    if has_audio and audio is not None:
        sr, arr = audio
        wf = torch.tensor(np.asarray(arr, dtype=np.float32))
        if wf.dim() > 1:
            wf = wf.mean(dim=-1)
        chunks = encoder.encode_waveform(wf, int(sr), max_chunks=max_audio_chunks)
        n_chunks = chunks.shape[0]
        inp = [VOCAB.env_mark] + [VOCAB.pad] * n_chunks
        audio_positions = torch.tensor(
            [list(range(1, 1 + n_chunks))], dtype=torch.long, device=dev,
        )
        audio_embeds = torch.zeros(1, n_chunks, model.config.dim, 2, device=dev)
        for j in range(n_chunks):
            feat = chunks[j:j + 1]
            z = model.project_audio(feat.unsqueeze(0)).squeeze(0)
            audio_embeds[0, j] = z[0]
    else:
        env = _words_to_token_ids(text_context or 'hello there', max_words=4)
        inp = [VOCAB.env_mark] + env

    input_ids = torch.tensor([inp], dtype=torch.long, device=dev)
    logits, _, _ = model(
        input_ids,
        audio_embeds=audio_embeds,
        audio_positions=audio_positions,
    )
    think_logits = logits[0, -1]
    think_ids = list(VOCAB.thinking_ids)
    sub = think_logits[think_ids]
    probs_t = F.softmax(sub, dim=-1)
    best_i = int(probs_t.argmax().item())
    best_id = think_ids[best_i]
    probs = {
        THINKING_NAMES[tid]: float(probs_t[i].item())
        for i, tid in enumerate(think_ids)
    }

    return DuplexPrediction(
        thinking=THINKING_NAMES[best_id],
        thinking_id=best_id,
        probs=probs,
        n_audio_chunks=n_chunks,
        has_audio=has_audio and audio is not None,
        checkpoint=checkpoint,
        preset=preset,
    )
