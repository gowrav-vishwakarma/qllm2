"""
Multi-timescale Complex Selective SSM with parallel scan.

Forked from V5, with explicit fast/medium/slow lane partitioning.
State dimensions are split into tiers with different decay rates so the
model has near-persistent channels for long-range coherence alongside
fast-decaying channels for local patterns.

State equation (all complex, diagonal A):
    h[t] = A_t * h[t-1] + B_t * x[t]
    y[t] = Re(C_t * h[t]) projected back to complex output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..init import InitStrategy

from .complex import (
    ComplexLinear, ComplexNorm, ModReLU,
    cmul, cabs, cnormalize, to_real,
)


@dataclass
class SSMState:
    hidden: torch.Tensor   # [num_layers, B, state_dim, 2]
    step: int


# ---------------------------------------------------------------------------
# Parallel prefix scan (unchanged from V5)
# ---------------------------------------------------------------------------

def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Parallel prefix scan for h[t] = a[t]*h[t-1] + b[t].
    Uses associative operator: (a1,b1) + (a2,b2) = (a1*a2, a2*b1+b2).
    """
    L = a.shape[-3]
    if L == 1:
        return b
    if L <= 32:
        return _sequential_scan(a, b)
    return _blelloch_scan(a, b)


def _sequential_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    L = a.shape[-3]
    h = b[..., 0:1, :, :]
    hs = [h]
    for t in range(1, L):
        h = cmul(a[..., t:t+1, :, :], h) + b[..., t:t+1, :, :]
        hs.append(h)
    return torch.cat(hs, dim=-3)


def _blelloch_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    L = a.shape[-3]
    next_pow2 = 1 << (L - 1).bit_length()
    if next_pow2 != L:
        pad = next_pow2 - L
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
    L = a.shape[-3]
    if L == 1:
        return b
    a_even = a[..., 0::2, :, :]
    a_odd = a[..., 1::2, :, :]
    b_even = b[..., 0::2, :, :]
    b_odd = b[..., 1::2, :, :]
    a_combined = cmul(a_even, a_odd)
    b_combined = cmul(a_odd, b_even) + b_odd
    h_half = _scan_pow2(a_combined, b_combined)
    h = torch.empty_like(b)
    h[..., 1::2, :, :] = h_half
    h[..., 0:1, :, :] = b[..., 0:1, :, :]
    if L > 2:
        h[..., 2::2, :, :] = cmul(a[..., 2::2, :, :], h_half[..., :-1, :, :]) + b[..., 2::2, :, :]
    return h


# ---------------------------------------------------------------------------
# Single SSM layer (multi-timescale aware)
# ---------------------------------------------------------------------------

class ComplexSSMLayer(nn.Module):
    """
    One layer of the complex selective SSM.

    Complex eigenvalues A = exp(log_decay + i*frequency) give:
    - |A| = exp(log_decay) < 1: exponential decay (memory length)
    - arg(A) = frequency: oscillation (timescale selectivity)

    Multi-timescale: log_A_real is initialized with tiered decay rates
    (fast/medium/slow lanes) so some state dimensions persist for thousands
    of tokens while others capture local patterns.

    TSO (Timescale-Separated Output): when enabled, each timescale tier
    gets its own C_proj and a learned gate selects which timescale to
    trust per position. This lets the model explicitly reason about
    "what do my fast/medium/slow states predict?" separately.

    GSP (Gated State Protection): when enabled, a learned gate per state
    dimension interpolates between "normal SSM update" and "freeze state".
    protect=1 -> A becomes identity, Bx becomes 0 (state preserved).
    protect=0 -> normal SSM dynamics. Parallel-scan compatible because
    the modified A'/Bx' is still a valid linear recurrence.
    """

    def __init__(
        self,
        dim: int,
        state_dim: int,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
        tso: bool = False,
        gsp: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.tso = tso
        self.gsp = gsp

        self.n_fast = int(state_dim * 0.4)
        self.n_medium = int(state_dim * 0.3)
        self.n_slow = state_dim - self.n_fast - self.n_medium

        if initializer is not None:
            log_real, log_imag = initializer.init_ssm_eigenvalues(state_dim)
            self.log_A_real = nn.Parameter(log_real)
            self.log_A_imag = nn.Parameter(log_imag)
            self.D = nn.Parameter(initializer.init_skip_connection(dim))
        else:
            log_real, log_imag = self._default_multiscale_init(state_dim)
            self.log_A_real = nn.Parameter(log_real)
            self.log_A_imag = nn.Parameter(log_imag)
            self.D = nn.Parameter(torch.randn(dim, 2) * 0.01)

        self.dt_proj = nn.Linear(dim * 2, state_dim)
        self.dt_bias = nn.Parameter(torch.zeros(state_dim) - 4.0)

        self.B_proj = ComplexLinear(dim, state_dim, bias=False, initializer=initializer)

        if tso:
            self.C_fast = ComplexLinear(self.n_fast, dim, bias=False, initializer=initializer)
            self.C_medium = ComplexLinear(self.n_medium, dim, bias=False, initializer=initializer)
            self.C_slow = ComplexLinear(self.n_slow, dim, bias=False, initializer=initializer)
            self.tso_gate = nn.Linear(dim, 3)
        else:
            self.C_proj = ComplexLinear(state_dim, dim, bias=False, initializer=initializer)

        if gsp:
            self.protect_gate = nn.Linear(dim, state_dim)
            nn.init.constant_(self.protect_gate.bias, -3.0)

        self.norm = ComplexNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _default_multiscale_init(state_dim: int):
        """Multi-timescale eigenvalue init: fast/medium/slow lanes."""
        n_fast = int(state_dim * 0.4)
        n_medium = int(state_dim * 0.3)
        n_slow = state_dim - n_fast - n_medium

        fast = torch.linspace(math.log(0.9), math.log(0.99), n_fast)
        medium = torch.linspace(math.log(0.999), math.log(0.9999), n_medium)
        slow = torch.linspace(math.log(0.99999), math.log(0.999999), n_slow)
        log_A_real = torch.cat([fast, medium, slow])

        log_A_imag = torch.linspace(0.001, math.pi, state_dim)
        return log_A_real, log_A_imag

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B_size, L, dim, _ = x.shape

        x_real_flat = to_real(x, 'concat')
        dt = F.softplus(self.dt_proj(x_real_flat) + self.dt_bias)

        decay = torch.exp(self.log_A_real)
        freq = self.log_A_imag

        A_mag = torch.exp(-dt * decay.unsqueeze(0).unsqueeze(0))
        A_real = A_mag * torch.cos(freq)
        A_imag = A_mag * torch.sin(freq)
        A = torch.stack([A_real, A_imag], dim=-1)

        Bx = self.B_proj(x)
        Bx = Bx * dt.unsqueeze(-1)

        if h0 is not None:
            first_Bx = cmul(A[:, 0:1], h0.unsqueeze(1)) + Bx[:, 0:1]
            Bx = torch.cat([first_Bx, Bx[:, 1:]], dim=1)

        if self.gsp:
            protect = torch.sigmoid(self.protect_gate(cabs(x)))
            protect = protect.unsqueeze(-1)
            identity = torch.zeros_like(A)
            identity[..., 0] = 1.0
            A = protect * identity + (1 - protect) * A
            Bx = (1 - protect) * Bx

        h = parallel_scan(A, Bx)

        if self.tso:
            h_fast = h[..., :self.n_fast, :]
            h_medium = h[..., self.n_fast:self.n_fast + self.n_medium, :]
            h_slow = h[..., self.n_fast + self.n_medium:, :]

            y_fast = self.C_fast(h_fast)
            y_medium = self.C_medium(h_medium)
            y_slow = self.C_slow(h_slow)

            gate = torch.softmax(self.tso_gate(cabs(x)), dim=-1)
            y = (gate[..., 0:1].unsqueeze(-1) * y_fast +
                 gate[..., 1:2].unsqueeze(-1) * y_medium +
                 gate[..., 2:3].unsqueeze(-1) * y_slow)
        else:
            y = self.C_proj(h)

        y = y + cmul(self.D.unsqueeze(0).unsqueeze(0).expand_as(x), x)

        if self.training:
            mask = self.dropout(torch.ones(B_size, L, dim, device=x.device))
            y = y * mask.unsqueeze(-1)

        return y, h[:, -1]


# ---------------------------------------------------------------------------
# Stacked SSM backbone
# ---------------------------------------------------------------------------

class ComplexSSM(nn.Module):
    """Stacked multi-timescale complex selective SSM."""

    def __init__(
        self,
        dim: int = 256,
        state_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.1,
        initializer: Optional['InitStrategy'] = None,
        tso: bool = False,
        gsp: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            ComplexSSMLayer(dim, state_dim, dropout, initializer=initializer, tso=tso, gsp=gsp)
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
        x: torch.Tensor,
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
