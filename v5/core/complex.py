"""
Complex algebraic primitives for V5.

Every operation preserves the algebraic structure of complex numbers.
Phase information flows through the entire network without degradation.

Key principle: NEVER apply real-valued activations to complex data.
Use modReLU (phase-preserving) instead of GELU/ReLU on real parts.

Representation: tensors of shape [..., dim, 2] where last dim is (real, imag).
We use manual real/imag storage rather than torch.complex64 for better
torch.compile compatibility and mixed-precision support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..init import InitStrategy


# ---------------------------------------------------------------------------
# Elementwise complex arithmetic (all JIT-scriptable)
# ---------------------------------------------------------------------------

@torch.jit.script
def cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex multiply: (a_r+i*a_i)(b_r+i*b_i)."""
    return torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
    ], dim=-1)


@torch.jit.script
def cconj(x: torch.Tensor) -> torch.Tensor:
    """Complex conjugate: (r, i) -> (r, -i)."""
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


@torch.jit.script
def cabs(x: torch.Tensor) -> torch.Tensor:
    """Magnitude |z| = sqrt(r^2 + i^2), shape [..., dim]."""
    return torch.sqrt(x[..., 0].square() + x[..., 1].square() + 1e-8)


@torch.jit.script
def cabs2(x: torch.Tensor) -> torch.Tensor:
    """Squared magnitude |z|^2, shape [..., dim]."""
    return x[..., 0].square() + x[..., 1].square()


@torch.jit.script
def cnormalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize to unit magnitude, preserving phase: z / |z|."""
    mag = cabs(x).unsqueeze(-1)  # [..., dim, 1]
    return x / mag


@torch.jit.script
def creal_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Re(a * conj(b)) summed over feature dim. Returns [...] scalar."""
    return (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).sum(dim=-1)


# ---------------------------------------------------------------------------
# modReLU -- the correct complex activation
# ---------------------------------------------------------------------------

class ModReLU(nn.Module):
    """
    Phase-preserving activation: threshold on magnitude, phase untouched.

        modReLU(z) = ReLU(|z| + b) * z/|z|      if |z| + b > 0
                     0                            otherwise

    The learnable bias `b` (one per feature) controls the dead-zone radius.
    Initialized slightly negative so most neurons are active at init.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), -0.1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., dim, 2] -> [..., dim, 2]"""
        mag = cabs(z)                                # [..., dim]
        activated_mag = F.relu(mag + self.bias)       # [..., dim]
        phase = z / (mag.unsqueeze(-1) + 1e-8)        # [..., dim, 2] unit phase
        return phase * activated_mag.unsqueeze(-1)     # [..., dim, 2]


# ---------------------------------------------------------------------------
# ComplexLinear -- complex matrix multiply
# ---------------------------------------------------------------------------

class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer: y = W @ x + b, all complex.

    W = W_r + i*W_i encodes rotation+scaling per weight.
    2 real params per weight, but each does the work of a 2x2 real matrix
    constrained to rotation+scaling.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if initializer is not None:
            wr, wi = initializer.init_complex_linear(out_dim, in_dim)
            self.weight_real = nn.Parameter(wr)
            self.weight_imag = nn.Parameter(wi)
        else:
            scale = (2 / (in_dim + out_dim)) ** 0.5
            self.weight_real = nn.Parameter(torch.randn(out_dim, in_dim) * scale)
            self.weight_imag = nn.Parameter(torch.randn(out_dim, in_dim) * scale)

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_dim))
            self.bias_imag = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim, 2] -> [..., out_dim, 2]"""
        xr, xi = x[..., 0], x[..., 1]

        yr = F.linear(xr, self.weight_real) - F.linear(xi, self.weight_imag)
        yi = F.linear(xr, self.weight_imag) + F.linear(xi, self.weight_real)

        if self.bias_real is not None:
            yr = yr + self.bias_real
            yi = yi + self.bias_imag

        return torch.stack([yr, yi], dim=-1)


# ---------------------------------------------------------------------------
# ComplexNorm -- magnitude normalization, phase preserved
# ---------------------------------------------------------------------------

class ComplexNorm(nn.Module):
    """
    Normalize magnitude distribution across features, preserve phase exactly.
    Like RMSNorm but for complex: scale by 1/RMS(|z|), then learnable gain.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., dim, 2] -> [..., dim, 2]"""
        mag = cabs(z)                                           # [..., dim]
        rms = torch.sqrt(mag.square().mean(dim=-1, keepdim=True) + self.eps)
        mag_normed = mag / rms                                   # [..., dim]
        mag_scaled = mag_normed * self.scale                     # [..., dim]
        phase = z / (mag.unsqueeze(-1) + 1e-8)                   # [..., dim, 2]
        return phase * mag_scaled.unsqueeze(-1)                  # [..., dim, 2]


# ---------------------------------------------------------------------------
# ComplexGatedUnit (CGU) -- the core algebraic block
# ---------------------------------------------------------------------------

class ComplexGatedUnit(nn.Module):
    """
    The fundamental computation block of V5.

    Like SwiGLU but fully complex-algebraic:
    - Gate magnitude sigma(|W_g z|) selects HOW MUCH  (real [0,1])
    - Gate phase (W_g z)/|W_g z| selects WHAT ROTATION (unit complex)
    - Up-projection + modReLU provides nonlinearity
    - Down-projection maps back to model dim

    One CGU gate does 2x the work of a real SwiGLU gate because it
    simultaneously filters AND transforms.
    """

    def __init__(
        self,
        dim: int,
        expand: int = 2,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        hidden = dim * expand
        self.gate_proj = ComplexLinear(dim, hidden, bias=False, initializer=initializer)
        self.up_proj = ComplexLinear(dim, hidden, bias=False, initializer=initializer)
        self.down_proj = ComplexLinear(hidden, dim, bias=False, initializer=initializer)
        self.act = ModReLU(hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., dim, 2] -> [..., dim, 2]"""
        gate = self.gate_proj(z)                        # [..., hidden, 2]
        gate_mag = torch.sigmoid(cabs(gate))            # [..., hidden] in [0,1]
        gate_phase = gate / (cabs(gate).unsqueeze(-1) + 1e-8)  # [..., hidden, 2] unit

        up = self.up_proj(z)                            # [..., hidden, 2]
        up = self.act(up)                               # phase-preserving activation

        # Complex gating: magnitude selects, phase rotates
        gated = cmul(gate_phase, up) * gate_mag.unsqueeze(-1)

        return self.down_proj(gated)                    # [..., dim, 2]


# ---------------------------------------------------------------------------
# ComplexEmbed -- token embedding into complex space
# ---------------------------------------------------------------------------

class ComplexEmbed(nn.Module):
    """
    Embed tokens into complex space with learned real and imaginary components.

    Phase MEANS something here: tokens with similar semantics but different
    roles (e.g. "happy" vs "sad") can share magnitude but differ in phase.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        padding_idx: Optional[int] = None,
        initializer: Optional['InitStrategy'] = None,
    ):
        super().__init__()
        self.dim = dim
        self.embed_real = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.embed_imag = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)

        if initializer is not None:
            r, i = initializer.init_embedding(vocab_size, dim)
            with torch.no_grad():
                self.embed_real.weight.copy_(r)
                self.embed_imag.weight.copy_(i)
        else:
            nn.init.normal_(self.embed_real.weight, std=0.02)
            nn.init.normal_(self.embed_imag.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [B, S] -> [B, S, dim, 2]"""
        r = self.embed_real(token_ids)
        i = self.embed_imag(token_ids)
        return torch.stack([r, i], dim=-1)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def to_real(z: torch.Tensor, mode: str = 'concat') -> torch.Tensor:
    """Convert [..., dim, 2] complex to real tensor for output heads."""
    if mode == 'concat':
        return torch.cat([z[..., 0], z[..., 1]], dim=-1)  # [..., dim*2]
    elif mode == 'magnitude':
        return cabs(z)                                      # [..., dim]
    elif mode == 'real':
        return z[..., 0]                                    # [..., dim]
    raise ValueError(f"Unknown mode: {mode}")
