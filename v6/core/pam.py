"""
Phase-Associative Memory (PAM).

Replaces the vector-state ComplexSSM with a matrix-state architecture.
The state is S_t \in C^{H x d x d}.
This solves the interference problem of HSB by providing O(d^2) capacity
per head, allowing multiple facts to be stored without colliding in a
single vector dimension.

State Update:
  S_t = \\gamma_t S_{t-1} + V_t \\otimes K_t^*
Retrieval:
  Y_t = S_t Q_t = V_t (K_t^* \\cdot Q_t)

The dot product K_t^* \\cdot Q_t natively computes attention via complex
phase interference (constructive/destructive). No softmax is needed.

GSP (Gated State Protection) is integrated directly into the decay \\gamma_t:
  \\gamma_t = exp(-dt) * (1 - p_t) + p_t
where p_t is the protect gate. When p_t=1, \\gamma_t=1 and the state is frozen.

For efficient training, we use the Dual Form (Attention form) which computes
the output in O(T^2) time without materializing the d x d matrix sequentially.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..init import InitStrategy

from .complex import ComplexLinear, ComplexNorm, cmul, cabs, to_real


@dataclass
class PAMState:
    """Recurrent state for Phase-Associative Memory (used in inference)."""
    matrix: torch.Tensor  # [num_layers, B, H, d, d, 2]
    step: int


class PhaseAssociativeLayer(nn.Module):
    """
    One layer of Phase-Associative Memory.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
        gsp: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.gsp = gsp

        # Projections
        self.q_proj = ComplexLinear(dim, self.inner_dim, bias=False, initializer=initializer)
        self.k_proj = ComplexLinear(dim, self.inner_dim, bias=False, initializer=initializer)
        self.v_proj = ComplexLinear(dim, self.inner_dim, bias=False, initializer=initializer)
        self.o_proj = ComplexLinear(self.inner_dim, dim, bias=False, initializer=initializer)

        # Data-dependent decay (dt)
        self.dt_proj = nn.Linear(dim * 2, num_heads)
        self.dt_bias = nn.Parameter(torch.zeros(num_heads) - 4.0)

        # GSP (protect gate)
        if gsp:
            self.protect_gate = nn.Linear(dim, num_heads)
            nn.init.constant_(self.protect_gate.bias, -3.0)

        self.norm = ComplexNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, dim, 2]
            state: [B, H, d, d, 2] optional recurrent state for inference
        Returns:
            y: [B, T, dim, 2]
            new_state: [B, H, d, d, 2]
        """
        B, T, _, _ = x.shape
        H = self.num_heads
        d = self.head_dim

        # 1. Projections
        q = self.q_proj(x).view(B, T, H, d, 2).transpose(1, 2)  # [B, H, T, d, 2]
        k = self.k_proj(x).view(B, T, H, d, 2).transpose(1, 2)  # [B, H, T, d, 2]
        v = self.v_proj(x).view(B, T, H, d, 2).transpose(1, 2)  # [B, H, T, d, 2]

        # 2. Decay and GSP
        x_real_flat = to_real(x, 'concat')  # [B, T, dim*2]
        dt = F.softplus(self.dt_proj(x_real_flat) + self.dt_bias)  # [B, T, H]
        dt = dt.transpose(1, 2)  # [B, H, T]

        if self.gsp:
            p = torch.sigmoid(self.protect_gate(cabs(x)))  # [B, T, H]
            p = p.transpose(1, 2)  # [B, H, T]
            gamma = torch.exp(-dt) * (1 - p) + p
            v_prime = v * (1 - p).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = torch.exp(-dt)
            v_prime = v

        # Training: Dual Form (O(T^2))
        if state is None and T > 1:
            # Decay matrix D[t, i] = prod_{j=i+1}^t gamma_j
            delta = -torch.log(gamma + 1e-8)  # [B, H, T]
            C = torch.cumsum(delta, dim=-1)   # [B, H, T]
            # D = exp(C_i - C_t) for i <= t
            D = torch.exp(C.unsqueeze(-2) - C.unsqueeze(-1))  # [B, H, T, T]
            mask = torch.tril(torch.ones(T, T, device=x.device))
            D = D * mask

            # Complex dot product W = Q @ K^*
            qr, qi = q[..., 0], q[..., 1]  # [B, H, T, d]
            kr, ki = k[..., 0], k[..., 1]  # [B, H, T, d]
            
            wr = qr @ kr.transpose(-1, -2) + qi @ ki.transpose(-1, -2)  # [B, H, T, T]
            wi = qi @ kr.transpose(-1, -2) - qr @ ki.transpose(-1, -2)  # [B, H, T, T]

            # Apply decay
            ar = wr * D
            ai = wi * D

            # Output Y = A @ V'
            vpr, vpi = v_prime[..., 0], v_prime[..., 1]  # [B, H, T, d]
            yr = ar @ vpr - ai @ vpi  # [B, H, T, d]
            yi = ar @ vpi + ai @ vpr  # [B, H, T, d]
            
            y = torch.stack([yr, yi], dim=-1)  # [B, H, T, d, 2]
            
            # We don't compute the full final state during training to save time
            # If needed for generation, it would be computed sequentially
            new_state = torch.empty(0, device=x.device)
            
        # Inference: Recurrent Form (O(T))
        else:
            if state is None:
                state = torch.zeros(B, H, d, d, 2, device=x.device, dtype=x.dtype)
            
            y_list = []
            S = state
            for t in range(T):
                v_t = v_prime[:, :, t].unsqueeze(-2)  # [B, H, d, 1, 2]
                k_t = k[:, :, t]
                k_t_conj = torch.stack([k_t[..., 0], -k_t[..., 1]], dim=-1).unsqueeze(-3)  # [B, H, 1, d, 2]
                
                # Outer product: v_t \otimes k_t^*
                outer_r = v_t[..., 0]*k_t_conj[..., 0] - v_t[..., 1]*k_t_conj[..., 1]
                outer_i = v_t[..., 0]*k_t_conj[..., 1] + v_t[..., 1]*k_t_conj[..., 0]
                outer = torch.stack([outer_r, outer_i], dim=-1)  # [B, H, d, d, 2]
                
                gamma_t = gamma[:, :, t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                S = S * gamma_t + outer
                
                q_t = q[:, :, t].unsqueeze(-3)  # [B, H, 1, d, 2]
                # S @ q_t
                sq_r = S[..., 0]*q_t[..., 0] - S[..., 1]*q_t[..., 1]
                sq_i = S[..., 0]*q_t[..., 1] + S[..., 1]*q_t[..., 0]
                y_t = torch.stack([sq_r, sq_i], dim=-1).sum(dim=-2)  # [B, H, d, 2]
                
                y_list.append(y_t)
                
            y = torch.stack(y_list, dim=2)  # [B, H, T, d, 2]
            new_state = S

        # 3. Output projection
        y = y.transpose(1, 2).contiguous().view(B, T, self.inner_dim, 2)
        out = self.o_proj(y)

        if self.training:
            mask = self.dropout(torch.ones(B, T, self.dim, device=x.device))
            out = out * mask.unsqueeze(-1)

        return out, new_state


class PhaseAssociativeMemory(nn.Module):
    """Stacked Phase-Associative Memory layers."""

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        head_dim: int = 64,
        num_layers: int = 16,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
        gsp: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            PhaseAssociativeLayer(dim, num_heads, head_dim, dropout, initializer=initializer, gsp=gsp)
            for _ in range(num_layers)
        ])
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1))
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            ComplexNorm(dim) for _ in range(num_layers)
        ])
        self.output_norm = ComplexNorm(dim)

    def init_state(self, batch_size: int, device: torch.device) -> PAMState:
        return PAMState(
            matrix=torch.zeros(
                self.num_layers, batch_size, self.num_heads, self.head_dim, self.head_dim, 2,
                device=device,
            ),
            step=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[PAMState] = None,
    ) -> Tuple[torch.Tensor, PAMState]:
        B, L, dim, _ = x.shape
        device = x.device

        if state is None and L == 1:
            state = self.init_state(B, device)

        h = x
        new_matrices = []

        for i, (layer, norm, scale) in enumerate(
            zip(self.layers, self.norms, self.layer_scales)
        ):
            s0 = state.matrix[i] if state is not None else None
            residual = h
            h_normed = norm(h)
            h_out, s_final = layer(h_normed, s0)
            h = residual + h_out * scale
            if state is not None:
                new_matrices.append(s_final)

        new_state = None
        if state is not None:
            new_state = PAMState(
                matrix=torch.stack(new_matrices, dim=0),
                step=state.step + L,
            )

        return self.output_norm(h), new_state
