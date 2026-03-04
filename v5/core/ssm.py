"""
Complex Selective SSM with parallel scan.

State equation (all complex, diagonal A):
    h[t] = A_t * h[t-1] + B_t * x[t]
    y[t] = Re(C_t * h[t]) projected back to complex output

Where A_t is input-dependent (selective) with complex diagonal entries
that naturally decompose into frequency bands for multi-timescale memory.

Parallel scan replaces V4's sequential loop, fixing the 17x slowdown.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

from .complex import (
    ComplexLinear, ComplexNorm, ModReLU,
    cmul, cabs, cnormalize, to_real,
)


@dataclass
class SSMState:
    hidden: torch.Tensor   # [num_layers, B, state_dim, 2]
    step: int


# ---------------------------------------------------------------------------
# Parallel prefix scan for complex diagonal recurrence
# ---------------------------------------------------------------------------

def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Parallel prefix scan for the recurrence h[t] = a[t]*h[t-1] + b[t].

    Uses the associative binary operator:
        (a1, b1) ⊕ (a2, b2) = (a1*a2, a2*b1 + b2)

    where * is elementwise complex multiplication.

    Args:
        a: [..., L, D, 2] complex diagonal coefficients (A_t)
        b: [..., L, D, 2] complex input terms (B_t @ x_t)

    Returns:
        h: [..., L, D, 2] complex hidden states at each timestep
    """
    L = a.shape[-3]

    if L == 1:
        return b

    # For short sequences, use sequential scan (less overhead)
    if L <= 32:
        return _sequential_scan(a, b)

    return _blelloch_scan(a, b)


def _sequential_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fallback sequential scan for short sequences."""
    L = a.shape[-3]
    h = b[..., 0:1, :, :]  # first step: h[0] = b[0] (no previous state)
    hs = [h]
    for t in range(1, L):
        h = cmul(a[..., t:t+1, :, :], h) + b[..., t:t+1, :, :]
        hs.append(h)
    return torch.cat(hs, dim=-3)


def _blelloch_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Blelloch-style parallel prefix scan. O(log L) sequential depth.

    For sequences that aren't a power of 2, we pad and then trim.
    """
    L = a.shape[-3]

    # Pad to power of 2
    next_pow2 = 1 << (L - 1).bit_length()
    if next_pow2 != L:
        pad = next_pow2 - L
        # Pad a with ones (identity for multiplication) and b with zeros
        a_pad = torch.ones_like(a[..., :1, :, :]).expand(
            *a.shape[:-3], pad, *a.shape[-2:]
        )
        b_pad = torch.zeros_like(b[..., :1, :, :]).expand(
            *b.shape[:-3], pad, *b.shape[-2:]
        )
        a = torch.cat([a, a_pad], dim=-3)
        b = torch.cat([b, b_pad], dim=-3)

    h = _scan_pow2(a, b)

    return h[..., :L, :, :]


def _scan_pow2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Parallel scan on power-of-2 length sequences."""
    L = a.shape[-3]
    if L == 1:
        return b

    # Up-sweep: combine pairs
    a_even = a[..., 0::2, :, :]
    a_odd = a[..., 1::2, :, :]
    b_even = b[..., 0::2, :, :]
    b_odd = b[..., 1::2, :, :]

    # (a_e, b_e) ⊕ (a_o, b_o) = (a_e * a_o, a_o * b_e + b_o)
    a_combined = cmul(a_even, a_odd)
    b_combined = cmul(a_odd, b_even) + b_odd

    # Recurse on half the length
    h_half = _scan_pow2(a_combined, b_combined)

    # Down-sweep: interleave results
    # Odd positions: h_half directly
    # Even positions: a[2k+1] * h_half[k-1] + b[2k] for k>0, b[0] for k=0
    h = torch.empty_like(b)

    # Odd positions (1, 3, 5, ...) = h_half results
    h[..., 1::2, :, :] = h_half

    # Even positions: h[2k] = a[2k+1]*h[2k-1] + b[2k] but h[2k-1] = h_half[k-1]
    # h[0] = b[0]
    h[..., 0:1, :, :] = b[..., 0:1, :, :]
    if L > 2:
        h[..., 2::2, :, :] = cmul(a[..., 2::2, :, :], h_half[..., :-1, :, :]) + b[..., 2::2, :, :]

    return h


# ---------------------------------------------------------------------------
# Single SSM layer
# ---------------------------------------------------------------------------

class ComplexSSMLayer(nn.Module):
    """
    One layer of the complex selective SSM.

    Complex eigenvalues A = exp(log_decay + i*frequency) give:
    - |A| = exp(log_decay) < 1: exponential decay (memory length)
    - arg(A) = frequency: oscillation (timescale selectivity)

    Input-dependent selectivity: the decay is modulated by the input,
    allowing content-dependent gating of what to remember/forget.
    """

    def __init__(
        self,
        dim: int,
        state_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim

        # Base complex eigenvalues
        # Decay: initialized so |A| ~ 0.95-0.999 (long memory)
        self.log_A_real = nn.Parameter(
            torch.linspace(math.log(0.95), math.log(0.999), state_dim)
        )
        # Frequency: log-spaced for multi-resolution decomposition
        self.log_A_imag = nn.Parameter(
            torch.linspace(0.001, math.pi, state_dim)
        )

        # Input-dependent modulation (selectivity): x -> dt (decay modifier)
        self.dt_proj = nn.Linear(dim * 2, state_dim)
        self.dt_bias = nn.Parameter(torch.zeros(state_dim) - 4.0)  # small dt initially

        # B: input -> state (complex)
        self.B_proj = ComplexLinear(dim, state_dim, bias=False)

        # C: state -> output (complex)
        self.C_proj = ComplexLinear(state_dim, dim, bias=False)

        # D: skip connection (complex)
        self.D = nn.Parameter(torch.randn(dim, 2) * 0.01)

        self.norm = ComplexNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,        # [B, L, dim, 2]
        h0: Optional[torch.Tensor] = None,  # [B, state_dim, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B_size, L, dim, _ = x.shape

        # Input-dependent dt (selectivity)
        x_real_flat = to_real(x, 'concat')             # [B, L, dim*2]
        dt = F.softplus(self.dt_proj(x_real_flat) + self.dt_bias)  # [B, L, state_dim]

        # Compute A_t = exp(-dt * exp(log_decay)) * exp(i * frequency)
        decay = torch.exp(self.log_A_real)              # [state_dim] in (0, 1)
        freq = self.log_A_imag                          # [state_dim]

        # A magnitude: decay modulated by dt
        A_mag = torch.exp(-dt * decay.unsqueeze(0).unsqueeze(0))  # [B, L, state_dim]

        # A phase: constant rotation per step
        A_real = A_mag * torch.cos(freq)                # [B, L, state_dim]
        A_imag = A_mag * torch.sin(freq)                # [B, L, state_dim]
        A = torch.stack([A_real, A_imag], dim=-1)       # [B, L, state_dim, 2]

        # B @ x: input projection to state space
        Bx = self.B_proj(x)                             # [B, L, state_dim, 2]

        # Scale by dt for discretization
        Bx = Bx * dt.unsqueeze(-1)

        # Prepend h0 contribution if provided
        if h0 is not None:
            # h[0] = A[0] * h0 + Bx[0]
            first_Bx = cmul(A[:, 0:1], h0.unsqueeze(1)) + Bx[:, 0:1]
            Bx = torch.cat([first_Bx, Bx[:, 1:]], dim=1)

        # Parallel scan: h = scan(A, Bx)
        h = parallel_scan(A, Bx)                        # [B, L, state_dim, 2]

        # Output: y = C @ h + D * x
        y = self.C_proj(h)                              # [B, L, dim, 2]
        y = y + cmul(self.D.unsqueeze(0).unsqueeze(0).expand_as(x), x)

        if self.training:
            mask = self.dropout(torch.ones(B_size, L, dim, device=x.device))
            y = y * mask.unsqueeze(-1)

        return y, h[:, -1]  # output, final state


# ---------------------------------------------------------------------------
# Full stacked SSM backbone
# ---------------------------------------------------------------------------

class ComplexSSM(nn.Module):
    """
    Stacked complex selective SSM.

    N layers of ComplexSSMLayer with residual connections and ComplexNorm.
    Replaces V4's OscillatorySSM with parallel scan for throughput.
    """

    def __init__(
        self,
        dim: int = 256,
        state_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            ComplexSSMLayer(dim, state_dim, dropout)
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

    def init_state(self, batch_size: int, device: torch.device) -> SSMState:
        return SSMState(
            hidden=torch.zeros(
                self.num_layers, batch_size, self.state_dim, 2,
                device=device,
            ),
            step=0,
        )

    def forward(
        self,
        x: torch.Tensor,                   # [B, L, dim, 2]
        state: Optional[SSMState] = None,
    ) -> Tuple[torch.Tensor, SSMState]:
        B, L, dim, _ = x.shape
        device = x.device

        if state is None:
            state = self.init_state(B, device)

        h = x
        new_hiddens = []

        for i, (layer, norm, scale) in enumerate(
            zip(self.layers, self.norms, self.layer_scales)
        ):
            h0 = state.hidden[i] if state.hidden.shape[0] > i else None
            residual = h
            h_normed = norm(h)
            h_out, h_final = layer(h_normed, h0)
            h = residual + h_out * scale
            new_hiddens.append(h_final)

        new_state = SSMState(
            hidden=torch.stack(new_hiddens, dim=0),
            step=state.step + L,
        )

        return self.output_norm(h), new_state
