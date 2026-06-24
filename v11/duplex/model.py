"""V11DuplexLM: subclass of V11LM with embedding-path forward (no shared edits)."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from v11.model import V11Config, V11LM
from v7.model import ComplexLinear, ComplexNorm


class V11DuplexLM(V11LM):
    """Duplex LM — reuses V11 blocks/head; adds optional audio embedding projection."""

    def __init__(self, cfg: V11Config, audio_feat_dim: int = 0):
        super().__init__(cfg)
        self.audio_feat_dim = audio_feat_dim
        if audio_feat_dim > 0:
            self.audio_proj = ComplexLinear(audio_feat_dim, cfg.dim)
            self.audio_norm = ComplexNorm(cfg.dim)
        else:
            self.audio_proj = None
            self.audio_norm = None

    def project_audio(self, features: torch.Tensor) -> torch.Tensor:
        """Real features [B, T, F] -> complex embeds [B, T, dim, 2] (zero imag init path)."""
        if self.audio_proj is None:
            raise RuntimeError("audio_feat_dim was 0 at init; cannot project audio")
        z = torch.stack([features, torch.zeros_like(features)], dim=-1)
        z = self.audio_proj(z)
        z = self.audio_norm(z)
        return z

    def forward_embeds(
        self,
        z: torch.Tensor,
        states: Optional[List] = None,
        step_offset: int = 0,
    ) -> Tuple[torch.Tensor, List, torch.Tensor]:
        """Run backbone from precomputed complex embeddings [B, T, dim, 2]."""
        if self.pos_embed is not None:
            z = self.pos_embed(z, step_offset=step_offset)
        z = self.embed_norm(z)
        new_states = []
        for i, block in enumerate(self.blocks):
            s = states[i] if states is not None else None
            z, new_s = block(z, pam_state=s, step_offset=step_offset)
            new_states.append(new_s)
        z = self.output_norm(z)
        lm = self.lm_head_norm(self.lm_head_proj(z))
        logits = (
            lm[..., 0] @ self.embed.embed_real.weight.T
            + lm[..., 1] @ self.embed.embed_imag.weight.T
        )
        aux = torch.tensor(0.0, device=z.device)
        return logits, new_states, aux

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List] = None,
        step_offset: int = 0,
        labels: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None,
    ):
        z = self.embed(input_ids)
        if audio_embeds is not None and audio_positions is not None:
            z = z.clone()
            for b in range(z.shape[0]):
                idx = audio_positions[b]
                if idx.numel() == 0:
                    continue
                n = min(idx.numel(), audio_embeds.shape[1])
                z[b, idx[:n]] = audio_embeds[b, :n]
        logits, new_states, aux = self.forward_embeds(z, states=states, step_offset=step_offset)
        return logits, new_states, aux

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Causal CE: logits[:, t] predicts labels[:, t+1]."""
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def thinking_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        thinking_ids: Tuple[int, ...],
    ) -> Tuple[float, int]:
        """Accuracy on thinking-token predictions only."""
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        think_set = set(thinking_ids)
        correct = 0
        total = 0
        for tid in thinking_ids:
            mask = shift_labels == tid
            if not mask.any():
                continue
            pred = shift_logits.argmax(dim=-1)
            correct += (pred[mask] == tid).sum().item()
            total += mask.sum().item()
        if total == 0:
            return 0.0, 0
        return correct / total, total
