"""
Holographic State Binding (HSB) -- SSM-native compositional memory.

Instead of maintaining separate registers (slot attention in disguise),
HSB injects holographic bindings directly into the SSM state and retrieves
them via complex conjugate unbinding from the state.

The key insight: cmul(key, value) creates a holographic binding that can
be stored in the SSM state. cconj-based retrieval from the state recovers
the bound value. Since the SSM state is already protected by GSP (Gated
State Protection), the bindings persist naturally.

This is SSM-native: bind/unbind happens INSIDE the SSM layer, not as a
separate module. Everything goes through the parallel scan. No sequential
loops, no O(T^2), no registers.

Bottleneck design: bindings happen in a lower-dimensional subspace
(bind_dim) to keep parameter cost reasonable. Two small ComplexLinear
projections scatter/gather between bind_dim and state_dim.

Mathematical basis (HRR):
  - Bind:   b = cmul(key, value)     -- encodes "key HAS value"
  - Unbind: v ~ cmul(query, conj(h)) -- recovers value from state

Novelty: No published work injects HRR bindings into a complex SSM's
state update or performs holographic unbinding from SSM state for output.
"""

import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING

from .complex import ComplexLinear, cabs

if TYPE_CHECKING:
    from ..init import InitStrategy


class HolographicBindHead(nn.Module):
    """
    Computes holographic binding vectors to inject into SSM state.

    Pipeline: x [B,T,dim,2] -> key,value in bind_dim -> cmul -> scatter to state_dim
    The result is ADDED to Bx before parallel_scan.
    """

    def __init__(
        self,
        dim: int,
        state_dim: int,
        bind_dim: int,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.key_proj = ComplexLinear(dim, bind_dim, bias=False, initializer=initializer)
        self.value_proj = ComplexLinear(dim, bind_dim, bias=False, initializer=initializer)
        self.scatter_proj = ComplexLinear(bind_dim, state_dim, bias=False, initializer=initializer)

        self.bind_gate = nn.Linear(dim, 1)
        nn.init.zeros_(self.bind_gate.weight)
        self.bind_gate.bias = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, dim, 2]
        Returns:
            bind_signal: [B, T, state_dim, 2] -- to be added to Bx
        """
        key = self.key_proj(x)      # [B, T, bind_dim, 2]
        value = self.value_proj(x)  # [B, T, bind_dim, 2]

        bound_r = key[..., 0] * value[..., 0] - key[..., 1] * value[..., 1]
        bound_i = key[..., 0] * value[..., 1] + key[..., 1] * value[..., 0]
        bound = torch.stack([bound_r, bound_i], dim=-1)  # [B, T, bind_dim, 2]

        scattered = self.scatter_proj(bound)  # [B, T, state_dim, 2]

        gate = torch.sigmoid(self.bind_gate(cabs(x)))  # [B, T, 1]
        return scattered * gate.unsqueeze(-1)


class HolographicUnbindHead(nn.Module):
    """
    Retrieves holographic bindings from SSM state.

    Pipeline: h [B,T,state_dim,2] -> gather to bind_dim -> cmul(query, conj(h_gathered)) -> project to dim
    The result is ADDED to the SSM output y.
    """

    def __init__(
        self,
        dim: int,
        state_dim: int,
        bind_dim: int,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.query_proj = ComplexLinear(dim, bind_dim, bias=False, initializer=initializer)
        self.gather_proj = ComplexLinear(state_dim, bind_dim, bias=False, initializer=initializer)
        self.out_proj = ComplexLinear(bind_dim, dim, bias=False, initializer=initializer)

        self.unbind_gate = nn.Linear(dim, 1)
        nn.init.zeros_(self.unbind_gate.weight)
        self.unbind_gate.bias = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, dim, 2]
            h: [B, T, state_dim, 2] -- SSM hidden states
        Returns:
            retrieved: [B, T, dim, 2] -- to be added to SSM output y
        """
        query = self.query_proj(x)       # [B, T, bind_dim, 2]
        h_bind = self.gather_proj(h)     # [B, T, bind_dim, 2]

        # cmul(query, conj(h_bind))
        h_conj_r = h_bind[..., 0]
        h_conj_i = -h_bind[..., 1]
        unbound_r = query[..., 0] * h_conj_r - query[..., 1] * h_conj_i
        unbound_i = query[..., 0] * h_conj_i + query[..., 1] * h_conj_r
        unbound = torch.stack([unbound_r, unbound_i], dim=-1)  # [B, T, bind_dim, 2]

        retrieved = self.out_proj(unbound)  # [B, T, dim, 2]

        gate = torch.sigmoid(self.unbind_gate(cabs(x)))  # [B, T, 1]
        return retrieved * gate.unsqueeze(-1)
