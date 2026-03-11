"""
Event-based episodic memory for V6.

Replaces token-wise WorkingMemory with a system that:
1. Scores each position for persistence salience (EventSalienceHead)
2. Pools salient spans into compact event representations
3. Writes event summaries to memory slots at boundaries
4. Reads via phase-coherence sparse top-k retrieval

The SSM hidden state acts as the temporary relational buffer.
Memory commits only happen when the salience head detects a stable
event worth persisting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..init import InitStrategy

from .complex import (
    ComplexLinear, ComplexNorm, cabs,
)


class EventSalienceHead(nn.Module):
    """Scores each token position for event persistence value.

    Uses the complex representation magnitude and local contrast to
    determine whether a position is part of an event worth storing.
    High salience = entity, fact, or relation anchor.
    Low salience = function word, filler, boilerplate.
    """

    def __init__(
        self,
        dim: int,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.score_proj = ComplexLinear(dim, 1, bias=False, initializer=initializer)
        self.score_bias = nn.Parameter(torch.tensor(-1.0))
        self.novelty_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, L, dim, 2] complex representations from the phase core

        Returns:
            salience: [B, L] per-position salience scores in (0, 1)
        """
        phase_score = cabs(self.score_proj(z)).squeeze(-1).squeeze(-1)

        mag = cabs(z)
        avg_mag = mag.mean(-1)  # [B, L]
        avg_mag_pool = avg_mag.unsqueeze(1)  # [B, 1, L] for avg_pool1d
        local_mean = F.avg_pool1d(
            avg_mag_pool, kernel_size=5, stride=1, padding=2,
        ).squeeze(1)  # [B, L]
        novelty = (avg_mag - local_mean) * self.novelty_scale

        salience = torch.sigmoid(phase_score + novelty + self.score_bias)
        return salience


class EpisodicMemory(nn.Module):
    """Event-based episodic memory with span-pooled writes.

    Instead of writing every token to a slot, this module:
    1. Uses EventSalienceHead to find salient positions
    2. Pools salient spans into compact event vectors
    3. Writes event summaries to slots only at boundary positions
    4. Reads via phase-coherence sparse retrieval
    """

    def __init__(
        self,
        dim: int,
        num_slots: int = 32,
        read_topk: int = 8,
        salience_threshold: float = 0.5,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.read_topk = read_topk
        self.salience_threshold = salience_threshold

        self.salience_head = EventSalienceHead(dim, initializer=initializer)

        self.event_key_proj = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.event_value_proj = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.read_query_proj = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.read_norm = ComplexNorm(dim)

        self.register_buffer('write_ptr', torch.tensor(0, dtype=torch.long))

    def _pool_events(
        self,
        z: torch.Tensor,
        salience: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool salient positions into event representations.

        Groups consecutive above-threshold positions into spans and
        averages them weighted by salience.

        Args:
            z: [B, L, dim, 2]
            salience: [B, L]

        Returns:
            event_vecs: [B, max_events, dim, 2] pooled event vectors
            event_mask: [B, max_events] which events are valid
        """
        B, L, dim, _ = z.shape
        S = self.num_slots

        events_list = []
        masks_list = []

        for b in range(B):
            sal = salience[b]
            above = sal > self.salience_threshold

            spans = []
            in_span = False
            span_start = 0
            for t in range(L):
                if above[t] and not in_span:
                    span_start = t
                    in_span = True
                elif not above[t] and in_span:
                    spans.append((span_start, t))
                    in_span = False
            if in_span:
                spans.append((span_start, L))

            batch_events = torch.zeros(S, dim, 2, device=z.device)
            batch_mask = torch.zeros(S, device=z.device)

            for i, (start, end) in enumerate(spans[:S]):
                span_sal = sal[start:end].unsqueeze(-1).unsqueeze(-1)
                span_z = z[b, start:end]
                weighted = (span_z * span_sal).sum(dim=0) / span_sal.sum().clamp(min=1e-8)
                batch_events[i] = weighted
                batch_mask[i] = 1.0

            events_list.append(batch_events)
            masks_list.append(batch_mask)

        return torch.stack(events_list), torch.stack(masks_list)

    def forward(
        self,
        z: torch.Tensor,
        slot_keys: Optional[torch.Tensor] = None,
        slot_values: Optional[torch.Tensor] = None,
        slot_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, L, dim, 2] representations from SSM
            slot_keys/values: [B, num_slots, dim, 2] or None
            slot_mask: [B, num_slots] or None

        Returns:
            retrieved: [B, L, dim, 2]
            new_keys, new_values: [B, num_slots, dim, 2]
            new_mask: [B, num_slots]
            salience: [B, L] per-position salience scores
        """
        B, L, dim, _ = z.shape
        S = self.num_slots
        device = z.device

        if slot_keys is None:
            slot_keys = torch.zeros(B, S, dim, 2, device=device)
            slot_values = torch.zeros(B, S, dim, 2, device=device)
            slot_mask = torch.zeros(B, S, device=device)
            self.write_ptr.fill_(0)

        salience = self.salience_head(z)

        event_vecs, event_valid = self._pool_events(z, salience)
        event_keys = self.event_key_proj(event_vecs)
        event_values = self.event_value_proj(event_vecs)

        new_keys = slot_keys.clone()
        new_values = slot_values.clone()
        new_mask = slot_mask.clone()

        ptr = self.write_ptr.item()
        n_events = int(event_valid.sum(dim=-1).max().item())
        n_events = min(n_events, S)

        for i in range(n_events):
            si = (ptr + i) % S
            valid = event_valid[:, i].unsqueeze(-1).unsqueeze(-1)
            new_keys[:, si] = valid * event_keys[:, i] + (1 - valid) * slot_keys[:, si]
            new_values[:, si] = valid * event_values[:, i] + (1 - valid) * slot_values[:, si]
            new_mask[:, si] = torch.clamp(
                slot_mask[:, si] + event_valid[:, i], max=1.0,
            )

        if n_events > 0:
            self.write_ptr.fill_((ptr + n_events) % S)

        queries = self.read_query_proj(z)
        q_r, q_i = queries[..., 0], queries[..., 1]
        k_r, k_i = new_keys[..., 0], new_keys[..., 1]

        dot = torch.bmm(q_r, k_r.transpose(1, 2)) + torch.bmm(q_i, k_i.transpose(1, 2))
        q_mag = torch.sqrt(q_r.square().sum(-1, keepdim=True) + q_i.square().sum(-1, keepdim=True) + 1e-8)
        k_mag = torch.sqrt(k_r.square().sum(-1, keepdim=True) + k_i.square().sum(-1, keepdim=True) + 1e-8).transpose(1, 2)
        scores = dot / (q_mag * k_mag + 1e-8)
        scores = scores.masked_fill(new_mask.unsqueeze(1).expand_as(scores) == 0, -1e9)

        k = min(self.read_topk, S) if self.read_topk > 0 else S
        if k < S:
            topk_scores, topk_idx = scores.topk(k, dim=-1)
            attn = F.softmax(topk_scores, dim=-1)
            topk_idx_v = topk_idx.unsqueeze(-1).expand(B, L, k, dim)
            v_r = torch.gather(new_values[..., 0].unsqueeze(1).expand(B, L, S, dim), 2, topk_idx_v)
            v_i = torch.gather(new_values[..., 1].unsqueeze(1).expand(B, L, S, dim), 2, topk_idx_v)
            v_r = (attn.unsqueeze(-1) * v_r).sum(dim=2)
            v_i = (attn.unsqueeze(-1) * v_i).sum(dim=2)
        else:
            attn = F.softmax(scores, dim=-1)
            v_r = torch.bmm(attn, new_values[..., 0])
            v_i = torch.bmm(attn, new_values[..., 1])

        retrieved = torch.stack([v_r, v_i], dim=-1)

        return self.read_norm(retrieved), new_keys, new_values, new_mask, salience
