"""V11DuplexLM: subclass of V11LM with embedding-path forward (no shared edits)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from v11.model import V11Config, V11LM
from v7.model import ComplexLinear, ComplexNorm


class FrameHeads(nn.Module):
    """Per-frame heads on the shared audio hidden (real+imag concat -> 2*dim).

    Phase 1: ctc_head (text alignment).
    Phase 2 (reserved): vad_head (speech/noise).
    Phase 3 (reserved): speaker_head (target/interferer).
    """

    def __init__(self, dim: int, n_text: int):
        super().__init__()
        self.n_text = n_text
        self.blank_id = n_text  # CTC blank index
        in_dim = dim * 2
        self.ctc_head = nn.Linear(in_dim, n_text + 1)
        # Reserved for Phase 2/3 — instantiate when training those objectives.
        self.vad_head: Optional[nn.Linear] = None
        self.speaker_head: Optional[nn.Linear] = None

    def ctc_logits(self, frame_hidden: torch.Tensor) -> torch.Tensor:
        """frame_hidden [B, T, 2*dim] -> logits [B, T, n_text+1]."""
        return self.ctc_head(frame_hidden)


def _hidden_to_real(hidden: torch.Tensor) -> torch.Tensor:
    """Complex hidden [B, T, dim, 2] -> real concat [B, T, 2*dim]."""
    return torch.cat([hidden[..., 0], hidden[..., 1]], dim=-1)


class V11DuplexLM(V11LM):
    """Duplex LM — reuses V11 blocks/head; adds optional audio embedding projection."""

    def __init__(self, cfg: V11Config, audio_feat_dim: int = 0, n_text: int = 32000):
        super().__init__(cfg)
        self.audio_feat_dim = audio_feat_dim
        self.n_text = n_text
        if audio_feat_dim > 0:
            self.audio_proj = ComplexLinear(audio_feat_dim, cfg.dim)
            self.audio_norm = ComplexNorm(cfg.dim)
            self.frame_heads = FrameHeads(cfg.dim, n_text)
            # Phase 3: optional target-speaker conditioning projection.
            self.cond_proj: Optional[ComplexLinear] = None
        else:
            self.audio_proj = None
            self.audio_norm = None
            self.frame_heads = None
            self.cond_proj = None

    def project_audio(
        self,
        features: torch.Tensor,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Real features [B, T, F] -> complex embeds [B, T, dim, 2].

        cond_vec (Phase 3): optional [B, cond_dim] speaker embedding; when
        cond_proj is wired, it is broadcast-added to every audio frame.
        """
        if self.audio_proj is None:
            raise RuntimeError("audio_feat_dim was 0 at init; cannot project audio")
        z = torch.stack([features, torch.zeros_like(features)], dim=-1)
        z = self.audio_proj(z)
        if cond_vec is not None and self.cond_proj is not None:
            c = self.cond_proj(
                torch.stack([cond_vec, torch.zeros_like(cond_vec)], dim=-1).unsqueeze(1)
            )
            z = z + c
        z = self.audio_norm(z)
        return z

    def forward_embeds(
        self,
        z: torch.Tensor,
        states: Optional[List] = None,
        step_offset: int = 0,
        return_hidden: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, List, torch.Tensor],
        Tuple[torch.Tensor, List, torch.Tensor, torch.Tensor],
    ]:
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
        if return_hidden:
            return logits, new_states, aux, z
        return logits, new_states, aux

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List] = None,
        step_offset: int = 0,
        labels: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        z = self.embed(input_ids)
        if audio_embeds is not None and audio_positions is not None:
            z = z.clone()
            for b in range(z.shape[0]):
                idx = audio_positions[b]
                if idx.numel() == 0:
                    continue
                valid = idx >= 0  # skip -1 padding slots; align embeds column->position
                if not bool(valid.any()):
                    continue
                emb = audio_embeds[b, :idx.numel()]
                z[b, idx[valid]] = emb[valid]
        out = self.forward_embeds(z, states=states, step_offset=step_offset,
                                  return_hidden=return_hidden)
        return out

    def gather_audio_frame_hidden(
        self,
        hidden: torch.Tensor,
        audio_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather post-norm hiddens at audio slots -> [B, max_T, 2*dim], lengths [B]."""
        b, max_slots = audio_positions.shape
        dim = hidden.shape[2]
        out = hidden.new_zeros(b, max_slots, dim, 2)
        lengths = torch.zeros(b, dtype=torch.long, device=hidden.device)
        for i in range(b):
            idx = audio_positions[i]
            valid = idx >= 0
            n = int(valid.sum().item())
            if n == 0:
                continue
            out[i, :n] = hidden[i, idx[valid]]
            lengths[i] = n
        return _hidden_to_real(out), lengths

    def ctc_log_probs(
        self,
        hidden: torch.Tensor,
        audio_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CTC log-probs [B, T, n_text+1] and per-sample input lengths."""
        if self.frame_heads is None:
            raise RuntimeError('frame_heads not initialized (audio_feat_dim=0)')
        frame_h, lengths = self.gather_audio_frame_hidden(hidden, audio_positions)
        logits = self.frame_heads.ctc_logits(frame_h)
        return F.log_softmax(logits, dim=-1), lengths

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

    def count_parameters(self) -> Dict[str, int]:
        out = super().count_parameters()
        audio_p = 0
        if self.audio_proj is not None:
            audio_p += sum(p.numel() for p in self.audio_proj.parameters())
        if self.audio_norm is not None:
            audio_p += sum(p.numel() for p in self.audio_norm.parameters())
        if self.frame_heads is not None:
            audio_p += sum(p.numel() for p in self.frame_heads.parameters())
        if self.cond_proj is not None:
            audio_p += sum(p.numel() for p in self.cond_proj.parameters())
        if audio_p:
            out['audio (proj+heads)'] = audio_p
            out['total'] = out['total'] + audio_p
        return out
