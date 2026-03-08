"""
Memory modules for V6.

WorkingMemory: per-sequence non-decaying scratchpad with learned write/read.
    - Runtime tensors (not nn.Parameter), reset per sequence
    - Write/read projections are nn.Module params, trained via backprop
    - Phase-coherence retrieval: Re(query * conj(key)) / (|q|*|k|)

InternalMemory: trained slots (nn.Parameter) for general language knowledge.
    - Keys and values updated during training, frozen at inference
    - Same phase-coherence retrieval as working memory

PersistentMemory: loadable tensor for per-user cross-session memory.
    - Same retrieval mechanism, loaded from disk, not part of model.parameters()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..init import InitStrategy

from .complex import (
    ComplexLinear, ComplexNorm,
    cmul, cconj, cabs, cnormalize, creal_dot,
)


def phase_coherence_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    """
    Phase-coherence retrieval: Re(query * conj(key)) / (|query| * |key|).

    Args:
        query: [..., dim, 2] complex vector(s)
        keys: [num_slots, dim, 2] complex vectors

    Returns:
        scores: [..., num_slots] similarity scores
    """
    # query: [..., dim, 2], keys: [S, dim, 2]
    # Re(query * conj(key)) = q_r*k_r + q_i*k_i, summed over dim
    q_r, q_i = query[..., 0], query[..., 1]  # [..., dim]
    k_r, k_i = keys[..., 0], keys[..., 1]    # [S, dim]

    # dot product: [..., S]
    dot = torch.einsum('...d,sd->...s', q_r, k_r) + torch.einsum('...d,sd->...s', q_i, k_i)

    # magnitudes
    q_mag = torch.sqrt((q_r.square() + q_i.square()).sum(dim=-1, keepdim=True) + 1e-8)  # [..., 1]
    k_mag = torch.sqrt((k_r.square() + k_i.square()).sum(dim=-1) + 1e-8)  # [S]

    scores = dot / (q_mag * k_mag.unsqueeze(0) + 1e-8) if keys.dim() == 2 else dot / (q_mag * k_mag + 1e-8)
    return scores


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------

class WorkingMemory(nn.Module):
    """
    Differentiable per-sequence scratchpad with learned write/read.

    Slots are runtime tensors, reset at the start of each sequence.
    Write/read projections are trained via backprop. The write gate
    is initialized negative so the model starts selective.

    Fully differentiable: uses soft addressing (no in-place ops) so
    gradients flow through the write decisions back to the token
    representations that triggered the writes.
    """

    def __init__(
        self,
        dim: int,
        num_slots: int = 64,
        gate_bias: float = -2.0,
        read_topk: int = 8,
        slot_decay: float = 0.95,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.read_topk = read_topk
        self.slot_decay = slot_decay

        self.write_gate_proj = ComplexLinear(dim, 1, bias=False, initializer=initializer)
        self.gate_bias = nn.Parameter(torch.tensor(gate_bias))
        self.write_key_proj = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.write_value_proj = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.read_query_proj = ComplexLinear(dim, dim, bias=False, initializer=initializer)

        self.read_norm = ComplexNorm(dim)

        self.register_buffer('write_ptr', torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        x: torch.Tensor,
        slot_keys: Optional[torch.Tensor] = None,
        slot_values: Optional[torch.Tensor] = None,
        slot_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, dim, 2]
            slot_keys/values: [B, num_slots, dim, 2] or None
            slot_mask: [B, num_slots] or None

        Returns:
            retrieved: [B, L, dim, 2]
            new_keys, new_values: [B, num_slots, dim, 2]
            new_mask: [B, num_slots]
        """
        B, L, dim, _ = x.shape
        device = x.device
        S = self.num_slots

        if slot_keys is None:
            slot_keys = torch.zeros(B, S, dim, 2, device=device)
            slot_values = torch.zeros(B, S, dim, 2, device=device)
            slot_mask = torch.zeros(B, S, device=device)
            self.write_ptr.fill_(0)
        else:
            slot_mask = slot_mask * self.slot_decay

        gate_raw = self.write_gate_proj(x)                              # [B, L, 1, 2]
        write_gates = torch.sigmoid(cabs(gate_raw) + self.gate_bias)    # [B, L, 1]
        write_keys = self.write_key_proj(x)                             # [B, L, dim, 2]
        write_values = self.write_value_proj(x)                         # [B, L, dim, 2]
        queries = self.read_query_proj(x)                               # [B, L, dim, 2]

        num_writes = min(L, S)
        gate_scores = write_gates.squeeze(-1)  # [B, L]

        _, top_indices = gate_scores.topk(num_writes, dim=-1)  # [B, num_writes]

        top_indices_exp = top_indices.unsqueeze(-1).unsqueeze(-1).expand(B, num_writes, dim, 2)
        selected_keys = torch.gather(write_keys, 1, top_indices_exp)     # [B, num_writes, dim, 2]
        selected_values = torch.gather(write_values, 1, top_indices_exp) # [B, num_writes, dim, 2]
        selected_gates = torch.gather(gate_scores, 1, top_indices).unsqueeze(-1).unsqueeze(-1)  # [B, num_writes, 1, 1]

        new_keys = slot_keys.clone()
        new_values = slot_values.clone()
        new_mask = slot_mask.clone()

        ptr = self.write_ptr.item()
        write_indices = [(ptr + i) % S for i in range(num_writes)]
        write_idx = torch.tensor(write_indices, device=device)

        for wi in range(num_writes):
            si = write_idx[wi]
            new_keys[:, si] = selected_gates[:, wi] * selected_keys[:, wi] + (1 - selected_gates[:, wi]) * slot_keys[:, si]
            new_values[:, si] = selected_gates[:, wi] * selected_values[:, wi] + (1 - selected_gates[:, wi]) * slot_values[:, si]
            new_mask[:, si] = torch.clamp(slot_mask[:, si] + selected_gates[:, wi].squeeze(-1).squeeze(-1), max=1.0)

        self.write_ptr.fill_((ptr + num_writes) % S)

        # READ: phase-coherence retrieval with top-k sparse attention
        q_r, q_i = queries[..., 0], queries[..., 1]       # [B, L, dim]
        k_r, k_i = new_keys[..., 0], new_keys[..., 1]     # [B, S, dim]

        dot = torch.bmm(q_r, k_r.transpose(1, 2)) + torch.bmm(q_i, k_i.transpose(1, 2))  # [B, L, S]
        q_mag = torch.sqrt(q_r.square().sum(-1, keepdim=True) + q_i.square().sum(-1, keepdim=True) + 1e-8)
        k_mag = torch.sqrt(k_r.square().sum(-1, keepdim=True) + k_i.square().sum(-1, keepdim=True) + 1e-8).transpose(1, 2)
        scores = dot / (q_mag * k_mag + 1e-8)

        scores = scores.masked_fill(new_mask.unsqueeze(1).expand_as(scores) == 0, -1e9)

        k = min(self.read_topk, S) if self.read_topk > 0 else S
        if k < S:
            topk_scores, topk_idx = scores.topk(k, dim=-1)       # [B, L, k]
            attn = F.softmax(topk_scores, dim=-1)                 # [B, L, k]
            topk_idx_v = topk_idx.unsqueeze(-1).expand(B, L, k, dim)
            v_r = torch.gather(new_values[..., 0].unsqueeze(1).expand(B, L, S, dim), 2, topk_idx_v)
            v_i = torch.gather(new_values[..., 1].unsqueeze(1).expand(B, L, S, dim), 2, topk_idx_v)
            v_r = (attn.unsqueeze(-1) * v_r).sum(dim=2)  # [B, L, dim]
            v_i = (attn.unsqueeze(-1) * v_i).sum(dim=2)
        else:
            attn = F.softmax(scores, dim=-1)  # [B, L, S]
            v_r = torch.bmm(attn, new_values[..., 0])
            v_i = torch.bmm(attn, new_values[..., 1])

        retrieved = torch.stack([v_r, v_i], dim=-1)  # [B, L, dim, 2]

        return self.read_norm(retrieved), new_keys, new_values, new_mask


# ---------------------------------------------------------------------------
# Internal Memory
# ---------------------------------------------------------------------------

class InternalMemory(nn.Module):
    """
    Trained memory slots: nn.Parameter key/value pairs for general knowledge.

    Keys and values are updated during training via backprop.
    At inference, they're frozen (as part of model weights).
    Retrieval uses the same phase-coherence mechanism as working memory.
    """

    def __init__(
        self,
        dim: int,
        num_slots: int = 128,
        read_topk: int = 8,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.read_topk = read_topk

        if initializer is not None:
            (kr, ki), (vr, vi) = initializer.init_internal_memory_slots(num_slots, dim)
            self.keys = nn.Parameter(torch.stack([kr, ki], dim=-1))   # [S, dim, 2]
            self.values = nn.Parameter(torch.stack([vr, vi], dim=-1)) # [S, dim, 2]
        else:
            self.keys = nn.Parameter(torch.randn(num_slots, dim, 2) * 0.02)
            self.values = nn.Parameter(torch.randn(num_slots, dim, 2) * 0.02)

        self.query_proj = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.norm = ComplexNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from internal memory for each token position.

        Args:
            x: [B, L, dim, 2] token representations

        Returns:
            retrieved: [B, L, dim, 2] retrieved general knowledge
        """
        B, L, dim, _ = x.shape
        S = self.num_slots
        query = self.query_proj(x)  # [B, L, dim, 2]

        q_r, q_i = query[..., 0], query[..., 1]  # [B, L, dim]
        k_r, k_i = self.keys[..., 0], self.keys[..., 1]  # [S, dim]

        dot = torch.einsum('bld,sd->bls', q_r, k_r) + torch.einsum('bld,sd->bls', q_i, k_i)
        q_mag = torch.sqrt(q_r.square().sum(-1, keepdim=True) + q_i.square().sum(-1, keepdim=True) + 1e-8)
        k_mag = torch.sqrt(k_r.square().sum(-1) + k_i.square().sum(-1) + 1e-8)  # [S]
        scores = dot / (q_mag * k_mag + 1e-8)  # [B, L, S]

        k = min(self.read_topk, S) if self.read_topk > 0 else S
        if k < S:
            topk_scores, topk_idx = scores.topk(k, dim=-1)        # [B, L, k]
            attn = F.softmax(topk_scores, dim=-1)                  # [B, L, k]
            topk_idx_v = topk_idx.unsqueeze(-1).expand(B, L, k, dim)
            vals_r = self.values[..., 0].unsqueeze(0).unsqueeze(0).expand(B, L, S, dim)
            vals_i = self.values[..., 1].unsqueeze(0).unsqueeze(0).expand(B, L, S, dim)
            v_r = torch.gather(vals_r, 2, topk_idx_v)  # [B, L, k, dim]
            v_i = torch.gather(vals_i, 2, topk_idx_v)
            v_r = (attn.unsqueeze(-1) * v_r).sum(dim=2)  # [B, L, dim]
            v_i = (attn.unsqueeze(-1) * v_i).sum(dim=2)
        else:
            attn = F.softmax(scores, dim=-1)  # [B, L, S]
            v_r = torch.einsum('bls,sd->bld', attn, self.values[..., 0])
            v_i = torch.einsum('bls,sd->bld', attn, self.values[..., 1])

        retrieved = torch.stack([v_r, v_i], dim=-1)  # [B, L, dim, 2]

        return self.norm(retrieved)


# ---------------------------------------------------------------------------
# Persistent Memory (external, per-user)
# ---------------------------------------------------------------------------

class PersistentMemoryStore:
    """
    External memory store for per-user cross-session memory.

    Not an nn.Module -- this is a container for tensors that participate
    in the forward pass but are not part of model.parameters().
    Serializable with torch.save/torch.load.
    """

    def __init__(self, dim: int, num_slots: int = 256, device: torch.device = None):
        self.dim = dim
        self.num_slots = num_slots
        dev = device or torch.device('cpu')
        self.keys = torch.zeros(num_slots, dim, 2, device=dev)
        self.values = torch.zeros(num_slots, dim, 2, device=dev)
        self.mask = torch.zeros(num_slots, device=dev)
        self.write_ptr = 0

    def write(self, key: torch.Tensor, value: torch.Tensor):
        """Write a key-value pair to the next slot."""
        self.keys[self.write_ptr] = key.detach()
        self.values[self.write_ptr] = value.detach()
        self.mask[self.write_ptr] = 1.0
        self.write_ptr = (self.write_ptr + 1) % self.num_slots

    def get_keys_values(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (keys, values, mask) for use in forward pass."""
        return self.keys, self.values, self.mask

    def state_dict(self) -> dict:
        return {
            'keys': self.keys.cpu(),
            'values': self.values.cpu(),
            'mask': self.mask.cpu(),
            'write_ptr': self.write_ptr,
            'dim': self.dim,
            'num_slots': self.num_slots,
        }

    @classmethod
    def load(cls, state: dict, device: torch.device = None) -> 'PersistentMemoryStore':
        store = cls(state['dim'], state['num_slots'], device)
        store.keys = state['keys'].to(device or torch.device('cpu'))
        store.values = state['values'].to(device or torch.device('cpu'))
        store.mask = state['mask'].to(device or torch.device('cpu'))
        store.write_ptr = state['write_ptr']
        return store

    def to(self, device: torch.device) -> 'PersistentMemoryStore':
        self.keys = self.keys.to(device)
        self.values = self.values.to(device)
        self.mask = self.mask.to(device)
        return self


class PersistentMemoryReader(nn.Module):
    """
    nn.Module that reads from an externally-provided PersistentMemoryStore.
    The store's tensors are passed into forward() -- they're not model params.
    """

    def __init__(
        self,
        dim: int,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.query_proj = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.norm = ComplexNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Read from persistent memory.

        Args:
            x: [B, L, dim, 2]
            keys: [num_slots, dim, 2]
            values: [num_slots, dim, 2]
            mask: [num_slots]

        Returns:
            retrieved: [B, L, dim, 2]
        """
        query = self.query_proj(x)  # [B, L, dim, 2]

        q_r, q_i = query[..., 0], query[..., 1]
        k_r, k_i = keys[..., 0], keys[..., 1]

        dot = torch.einsum('bld,sd->bls', q_r, k_r) + torch.einsum('bld,sd->bls', q_i, k_i)
        q_mag = torch.sqrt(q_r.square().sum(-1, keepdim=True) + q_i.square().sum(-1, keepdim=True) + 1e-8)
        k_mag = torch.sqrt(k_r.square().sum(-1) + k_i.square().sum(-1) + 1e-8)
        scores = dot / (q_mag * k_mag + 1e-8)

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        v_r = torch.einsum('bls,sd->bld', attn, values[..., 0])
        v_i = torch.einsum('bls,sd->bld', attn, values[..., 1])
        retrieved = torch.stack([v_r, v_i], dim=-1)

        return self.norm(retrieved)


# ---------------------------------------------------------------------------
# Session Memory (optional, between-turn buffer)
# ---------------------------------------------------------------------------

class SessionMemoryBuffer:
    """
    Between-turn memory buffer for multi-turn conversations.

    Not an nn.Module. Accumulates context across conversation turns.
    Discarded at session end, or compressed into persistent memory.

    Only used when --session_memory is enabled.
    """

    def __init__(self, dim: int, num_slots: int = 128, device: torch.device = None):
        self.dim = dim
        self.num_slots = num_slots
        dev = device or torch.device('cpu')
        self.keys = torch.zeros(num_slots, dim, 2, device=dev)
        self.values = torch.zeros(num_slots, dim, 2, device=dev)
        self.mask = torch.zeros(num_slots, device=dev)
        self.write_ptr = 0

    def write(self, key: torch.Tensor, value: torch.Tensor):
        self.keys[self.write_ptr] = key.detach()
        self.values[self.write_ptr] = value.detach()
        self.mask[self.write_ptr] = 1.0
        self.write_ptr = (self.write_ptr + 1) % self.num_slots

    def write_batch(self, keys: torch.Tensor, values: torch.Tensor, importance: torch.Tensor):
        """Write multiple entries filtered by importance scores."""
        top_k = min(len(importance), self.num_slots - int(self.mask.sum()))
        if top_k <= 0:
            return
        _, top_idx = importance.topk(top_k)
        for idx in top_idx:
            self.write(keys[idx], values[idx])

    def get_keys_values(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.keys, self.values, self.mask

    def compress_to_persistent(self, persistent: PersistentMemoryStore, top_k: int = 64):
        """Transfer most important session entries to persistent memory."""
        if self.mask.sum() == 0:
            return
        active = self.mask > 0
        active_keys = self.keys[active]
        active_values = self.values[active]
        magnitudes = torch.sqrt(
            active_keys[..., 0].square().sum(-1) + active_keys[..., 1].square().sum(-1) + 1e-8
        )
        k = min(top_k, len(magnitudes))
        _, top_idx = magnitudes.topk(k)
        for idx in top_idx:
            persistent.write(active_keys[idx], active_values[idx])

    def clear(self):
        self.keys.zero_()
        self.values.zero_()
        self.mask.zero_()
        self.write_ptr = 0

    def to(self, device: torch.device) -> 'SessionMemoryBuffer':
        self.keys = self.keys.to(device)
        self.values = self.values.to(device)
        self.mask = self.mask.to(device)
        return self


# ---------------------------------------------------------------------------
# Expert Memory (read-only, shared)
# ---------------------------------------------------------------------------

class ExpertMemoryStore:
    """
    Read-only expert memory for domain-specific knowledge.

    Same tensor format as PersistentMemoryStore but loaded once and shared
    across users. Created by training or curating domain knowledge.
    """

    def __init__(self, dim: int, num_slots: int = 256, device: torch.device = None):
        self.dim = dim
        self.num_slots = num_slots
        dev = device or torch.device('cpu')
        self.keys = torch.zeros(num_slots, dim, 2, device=dev)
        self.values = torch.zeros(num_slots, dim, 2, device=dev)
        self.mask = torch.zeros(num_slots, device=dev)

    def get_keys_values(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.keys, self.values, self.mask

    def state_dict(self) -> dict:
        return {
            'keys': self.keys.cpu(),
            'values': self.values.cpu(),
            'mask': self.mask.cpu(),
            'dim': self.dim,
            'num_slots': self.num_slots,
        }

    @classmethod
    def load(cls, state: dict, device: torch.device = None) -> 'ExpertMemoryStore':
        store = cls(state['dim'], state['num_slots'], device)
        store.keys = state['keys'].to(device or torch.device('cpu'))
        store.values = state['values'].to(device or torch.device('cpu'))
        store.mask = state['mask'].to(device or torch.device('cpu'))
        return store

    @classmethod
    def from_persistent(cls, persistent: PersistentMemoryStore) -> 'ExpertMemoryStore':
        """Convert a persistent memory into a read-only expert memory."""
        store = cls(persistent.dim, persistent.num_slots)
        store.keys = persistent.keys.clone()
        store.values = persistent.values.clone()
        store.mask = persistent.mask.clone()
        return store

    def to(self, device: torch.device) -> 'ExpertMemoryStore':
        self.keys = self.keys.to(device)
        self.values = self.values.to(device)
        self.mask = self.mask.to(device)
        return self


class ExpertMemoryReader(nn.Module):
    """Reads from an externally-provided ExpertMemoryStore. Same interface as PersistentMemoryReader."""

    def __init__(self, dim: int, initializer: Optional['InitStrategy'] = None):
        super().__init__()
        self.query_proj = ComplexLinear(dim, dim, bias=False, initializer=initializer)
        self.norm = ComplexNorm(dim)

    def forward(self, x, keys, values, mask):
        query = self.query_proj(x)
        q_r, q_i = query[..., 0], query[..., 1]
        k_r, k_i = keys[..., 0], keys[..., 1]

        dot = torch.einsum('bld,sd->bls', q_r, k_r) + torch.einsum('bld,sd->bls', q_i, k_i)
        q_mag = torch.sqrt(q_r.square().sum(-1, keepdim=True) + q_i.square().sum(-1, keepdim=True) + 1e-8)
        k_mag = torch.sqrt(k_r.square().sum(-1) + k_i.square().sum(-1) + 1e-8)
        scores = dot / (q_mag * k_mag + 1e-8)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        v_r = torch.einsum('bls,sd->bld', attn, values[..., 0])
        v_i = torch.einsum('bls,sd->bld', attn, values[..., 1])
        return self.norm(torch.stack([v_r, v_i], dim=-1))


# ---------------------------------------------------------------------------
# Adaptation: persistent memory as soft training signal
# ---------------------------------------------------------------------------

class MemoryAdaptation:
    """
    Uses persistent memory as a soft training signal for personalization.

    When a user corrects the model or shows preference, high-coherence
    patterns in persistent memory nudge internal memory toward user preferences.
    Phase-space distillation from user feedback -- no separate RLHF pipeline.
    """

    @staticmethod
    def compute_adaptation_loss(
        internal_memory: InternalMemory,
        persistent_keys: torch.Tensor,
        persistent_values: torch.Tensor,
        persistent_mask: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Soft alignment loss between internal and persistent memory.

        For each internal memory slot, find the best matching persistent
        slot via phase coherence, then nudge internal values toward
        persistent values for high-coherence pairs.
        """
        if persistent_mask.sum() == 0:
            return torch.tensor(0.0, device=internal_memory.keys.device)

        im_k_r = internal_memory.keys[..., 0]
        im_k_i = internal_memory.keys[..., 1]
        pm_k_r = persistent_keys[..., 0]
        pm_k_i = persistent_keys[..., 1]

        dot = torch.mm(im_k_r, pm_k_r.T) + torch.mm(im_k_i, pm_k_i.T)
        im_mag = torch.sqrt(im_k_r.square().sum(-1, keepdim=True) + im_k_i.square().sum(-1, keepdim=True) + 1e-8)
        pm_mag = torch.sqrt(pm_k_r.square().sum(-1, keepdim=True) + pm_k_i.square().sum(-1, keepdim=True) + 1e-8)
        coherence = dot / (im_mag * pm_mag.T + 1e-8)
        coherence = coherence * persistent_mask.unsqueeze(0)

        weights = F.softmax(coherence / temperature, dim=-1)

        target_r = torch.mm(weights, persistent_values[..., 0])
        target_i = torch.mm(weights, persistent_values[..., 1])

        im_v_r = internal_memory.values[..., 0]
        im_v_i = internal_memory.values[..., 1]

        return (
            (im_v_r - target_r.detach()).square().mean() +
            (im_v_i - target_i.detach()).square().mean()
        )
