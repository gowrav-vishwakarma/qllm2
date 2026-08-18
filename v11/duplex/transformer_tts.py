"""Matched GPT-2-style transformer backbone for the duplex TTS task.

Same unified vocab and Mimi delay-pattern sequences as V11DuplexLM, so PAM vs
transformer is an apples-to-apples codec next-token comparison. Geometry is the
v6 ~100M transformer with the duplex 40k vocab (~93M, matching duplex_100m
without the ASR audio/CTC heads).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from v6.transformer_baseline import TransformerConfig, TransformerLM


def transformer_tts_config(vocab_size: int, max_seq_len: int = 2048) -> TransformerConfig:
    """~93M params at vocab=40208 (duplex_100m tied-embed budget)."""
    return TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=672,
        n_layers=12,
        n_heads=12,
        d_ff=2688,
        dropout=0.1,
        tie_weights=True,
    )


class TransformerTTS(nn.Module):
    """Thin wrapper so train_tts can share the PAM loss / generate loop.

    `forward` returns `(logits, states, aux)` like V11DuplexLM. Recurrent state
    is unused — each call re-encodes the full sequence (KV cache is the thing
    PAM is supposed to beat on latency, not a training requirement).
    """

    supports_recurrent = False

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 2048,
        config: Optional[TransformerConfig] = None,
    ):
        super().__init__()
        self.config = config or transformer_tts_config(vocab_size, max_seq_len=max_seq_len)
        self.lm = TransformerLM(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List] = None,
        step_offset: int = 0,
        labels: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        del states, step_offset, labels, audio_embeds, audio_positions, return_hidden
        T = input_ids.size(1)
        max_t = self.config.max_seq_len
        if T > max_t:
            input_ids = input_ids[:, -max_t:]
        logits = self.lm(input_ids)
        aux = torch.tensor(0.0, device=input_ids.device)
        return logits, None, aux

    def count_parameters(self) -> Dict[str, int]:
        return self.lm.count_parameters()
