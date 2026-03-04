"""
Structured initialization strategies for V5 complex-valued architecture.

Each strategy is a grokking attractor: structured patterns that weight decay
pulls the model toward during training, potentially inducing generalization.

Strategies are selectable via --init_strategy and reproducible via --init_seed.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import torch


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class InitStrategy(ABC):
    """Abstract base for initialization strategies. All use seed for reproducibility."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self._gen = torch.Generator().manual_seed(self.seed)

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Init", "").lower()

    def _scale(self, in_dim: int, out_dim: int) -> float:
        """Xavier-style scale for complex linear layers."""
        return (2 / (in_dim + out_dim)) ** 0.5

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (weight_real, weight_imag) each shape [out_dim, in_dim]."""
        scale = self._scale(in_dim, out_dim)
        w = torch.randn(out_dim, in_dim, generator=self._gen, device=device) * scale
        return w.clone(), torch.randn(out_dim, in_dim, generator=self._gen, device=device).mul_(scale)

    def init_ssm_eigenvalues(
        self, state_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (log_A_real, log_A_imag). log_A_real in log(0.95)..log(0.999), log_A_imag in 0.001..pi."""
        log_A_real = torch.linspace(math.log(0.95), math.log(0.999), state_dim, device=device)
        log_A_imag = torch.linspace(0.001, math.pi, state_dim, device=device)
        return log_A_real, log_A_imag

    def init_phase_rotation(self, dim: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return [dim, 2] near identity (real=1, imag~0)."""
        rot = torch.zeros(dim, 2, device=device)
        rot[:, 0] = 1.0
        rot[:, 1] = torch.randn(dim, generator=self._gen, device=device) * 0.01
        return rot

    def init_embedding(
        self, vocab_size: int, dim: int, std: float = 0.02, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (embed_real, embed_imag) each shape [vocab_size, dim]."""
        r = torch.randn(vocab_size, dim, generator=self._gen, device=device) * std
        i = torch.randn(vocab_size, dim, generator=self._gen, device=device) * std
        return r, i

    def init_skip_connection(
        self, dim: int, std: float = 0.01, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Return [dim, 2] for D skip connection."""
        return torch.randn(dim, 2, generator=self._gen, device=device) * std

    def get_info(self) -> dict:
        return {"strategy": self.name, "seed": self.seed}


# ---------------------------------------------------------------------------
# Category 1: Random variants
# ---------------------------------------------------------------------------

class RandomInit(InitStrategy):
    """Normal random (current default), with explicit seed."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        wr = torch.randn(out_dim, in_dim, generator=self._gen, device=device) * scale
        wi = torch.randn(out_dim, in_dim, generator=self._gen, device=device) * scale
        return wr, wi


class UniformInit(InitStrategy):
    """Uniform in [-scale, scale]."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        wr = (torch.rand(out_dim, in_dim, generator=self._gen, device=device) * 2 - 1) * scale
        wi = (torch.rand(out_dim, in_dim, generator=self._gen, device=device) * 2 - 1) * scale
        return wr, wi


class OrthogonalInit(InitStrategy):
    """Random orthogonal via QR of random matrix."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        m = max(out_dim, in_dim)
        qr = torch.randn(m, m, generator=self._gen, device=device)
        q, _ = torch.linalg.qr(qr)
        wr = q[:out_dim, :in_dim] * scale
        qr2 = torch.randn(m, m, generator=self._gen, device=device)
        q2, _ = torch.linalg.qr(qr2)
        wi = q2[:out_dim, :in_dim] * scale
        return wr, wi


# ---------------------------------------------------------------------------
# Category 2: Number-theoretic / quasi-random
# ---------------------------------------------------------------------------

def _weyl_sequence(n: int, alpha: float) -> torch.Tensor:
    """Weyl sequence: frac(k * alpha) for k=0..n-1."""
    return torch.tensor([(k * alpha) % 1.0 for k in range(n)], dtype=torch.float32)


class GoldenRatioInit(InitStrategy):
    """Golden angle (2pi/phi^2) spacing; Weyl sequence for magnitudes."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = (1 + math.sqrt(5)) / 2
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        angles = _weyl_sequence(n, 1 / phi)
        mags = _weyl_sequence(n, phi)
        mags = (mags * 2 - 1) * scale
        wr = (mags * torch.cos(angles * 2 * math.pi)).view(out_dim, in_dim)
        wi = (mags * torch.sin(angles * 2 * math.pi)).view(out_dim, in_dim)
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class PiInit(InitStrategy):
    """Weyl sequence using pi."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        angles_r = _weyl_sequence(n, math.pi)
        angles_i = _weyl_sequence(n, math.e)
        mags = (angles_r * 2 - 1) * scale
        wr = (mags * torch.cos(angles_i * 2 * math.pi)).view(out_dim, in_dim)
        wi = (mags * torch.sin(angles_i * 2 * math.pi)).view(out_dim, in_dim)
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class SqrtPrimesInit(InitStrategy):
    """Weyl from sqrt(2), sqrt(3), sqrt(5)..."""

    _primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        p = self._primes
        angles_r = _weyl_sequence(n, math.sqrt(p[0]))
        angles_i = _weyl_sequence(n, math.sqrt(p[1]))
        mags = (angles_r * 2 - 1) * scale
        wr = (mags * torch.cos(angles_i * 2 * math.pi)).view(out_dim, in_dim)
        wi = (mags * torch.sin(angles_i * 2 * math.pi)).view(out_dim, in_dim)
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


# ---------------------------------------------------------------------------
# Category 3: Trigonometric
# ---------------------------------------------------------------------------

class SinusoidalInit(InitStrategy):
    """sin/cos at geometric frequencies (Vaswani-style)."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        positions = torch.arange(out_dim * in_dim, dtype=torch.float32)
        dim_scale = 10000.0 ** (torch.arange(2, dtype=torch.float32) / 2)
        angles = positions.unsqueeze(-1) / dim_scale
        wr = (scale * torch.sin(angles[:, 0])).view(out_dim, in_dim)
        wi = (scale * torch.cos(angles[:, 1])).view(out_dim, in_dim)
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class CircularInit(InitStrategy):
    """Uniform angles on concentric rings with radius decay."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        idx = torch.arange(n, dtype=torch.float32)
        r = scale * (1 - idx / (n + 1))
        theta = (idx * 2 * math.pi * (1 + math.sqrt(5)) / 2) % (2 * math.pi)
        wr = (r * torch.cos(theta)).view(out_dim, in_dim)
        wi = (r * torch.sin(theta)).view(out_dim, in_dim)
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class RootsOfUnityInit(InitStrategy):
    """N-th roots of unity cycling through rows."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        k = torch.arange(n, dtype=torch.float32)
        theta = 2 * math.pi * k / max(n, 1)
        vals = scale * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        wr = vals[:, 0].view(out_dim, in_dim)
        wi = vals[:, 1].view(out_dim, in_dim)
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


# ---------------------------------------------------------------------------
# Category 4: Spiral / nature-inspired
# ---------------------------------------------------------------------------

class FibonacciSpiralInit(InitStrategy):
    """Golden angle spiral (phyllotaxis): angle = n * 137.5 deg, radius = sqrt(n)."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        golden_angle = math.pi * (3 - math.sqrt(5))
        idx = torch.arange(n, dtype=torch.float32)
        theta = idx * golden_angle
        r = scale * torch.sqrt(idx + 1)
        wr = (r * torch.cos(theta)).view(out_dim, in_dim)
        wi = (r * torch.sin(theta)).view(out_dim, in_dim)
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class LogSpiralInit(InitStrategy):
    """Logarithmic spiral: r = a * e^(b*theta)."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        idx = torch.arange(n, dtype=torch.float32)
        theta = idx * 0.5
        r = scale * torch.exp(0.1 * theta)
        wr = (r * torch.cos(theta)).view(out_dim, in_dim)
        wi = (r * torch.sin(theta)).view(out_dim, in_dim)
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class FermatSpiralInit(InitStrategy):
    """Fermat spiral: r = sqrt(theta)."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        idx = torch.arange(n, dtype=torch.float32) + 1
        theta = idx * 0.3
        r = scale * torch.sqrt(theta)
        wr = (r * torch.cos(theta)).view(out_dim, in_dim)
        wi = (r * torch.sin(theta)).view(out_dim, in_dim)
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


# ---------------------------------------------------------------------------
# Category 5: Fourier / spectral
# ---------------------------------------------------------------------------

class DFTInit(InitStrategy):
    """Discrete Fourier Transform basis rows."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = min(out_dim, in_dim)
        k = torch.arange(n, dtype=torch.float32)
        j = torch.arange(n, dtype=torch.float32)
        theta = 2 * math.pi * k.unsqueeze(1) * j.unsqueeze(0) / n
        wr = scale * torch.cos(theta)
        wi = scale * torch.sin(theta)
        if out_dim > n or in_dim > n:
            wr = torch.nn.functional.pad(wr, (0, max(0, in_dim - n), 0, max(0, out_dim - n)))
            wi = torch.nn.functional.pad(wi, (0, max(0, in_dim - n), 0, max(0, out_dim - n)))
            extra_r = torch.randn(out_dim, in_dim, generator=self._gen) * scale * 0.1
            extra_i = torch.randn(out_dim, in_dim, generator=self._gen) * scale * 0.1
            wr = wr + extra_r
            wi = wi + extra_i
        wr = wr[:out_dim, :in_dim]
        wi = wi[:out_dim, :in_dim]
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class DCTInit(InitStrategy):
    """Discrete Cosine Transform basis."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = min(out_dim, in_dim)
        k = torch.arange(n, dtype=torch.float32)
        j = torch.arange(n, dtype=torch.float32)
        wr = scale * torch.cos(math.pi * k.unsqueeze(1) * (j.unsqueeze(0) + 0.5) / n)
        wi = scale * torch.cos(math.pi * (k.unsqueeze(1) + 0.5) * j.unsqueeze(0) / n)
        if out_dim > n or in_dim > n:
            wr = torch.nn.functional.pad(wr, (0, max(0, in_dim - n), 0, max(0, out_dim - n)))
            wi = torch.nn.functional.pad(wi, (0, max(0, in_dim - n), 0, max(0, out_dim - n)))
            extra = torch.randn(out_dim, in_dim, generator=self._gen) * scale * 0.1
            wr = wr + extra
            wi = wi + extra
        wr = wr[:out_dim, :in_dim]
        wi = wi[:out_dim, :in_dim]
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class HartleyInit(InitStrategy):
    """Hartley: cas(k) = cos(k) + sin(k)."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = min(out_dim, in_dim)
        k = torch.arange(n, dtype=torch.float32)
        j = torch.arange(n, dtype=torch.float32)
        theta = 2 * math.pi * k.unsqueeze(1) * j.unsqueeze(0) / n
        cas = torch.cos(theta) + torch.sin(theta)
        wr = scale * cas
        wi = scale * torch.cos(theta)
        if out_dim > n or in_dim > n:
            wr = torch.nn.functional.pad(wr, (0, max(0, in_dim - n), 0, max(0, out_dim - n)))
            wi = torch.nn.functional.pad(wi, (0, max(0, in_dim - n), 0, max(0, out_dim - n)))
            extra = torch.randn(out_dim, in_dim, generator=self._gen) * scale * 0.1
            wr = wr + extra
            wi = wi + extra
        wr = wr[:out_dim, :in_dim]
        wi = wi[:out_dim, :in_dim]
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


# ---------------------------------------------------------------------------
# Category 6: Low-discrepancy
# ---------------------------------------------------------------------------

def _van_der_corput(n: int, base: int = 2) -> torch.Tensor:
    out = []
    for i in range(n):
        v, b, f = 0.0, 1.0 / base, 1
        k = i + 1
        while k:
            f *= base
            v += (k % base) * b
            b /= base
            k //= base
        out.append(v)
    return torch.tensor(out, dtype=torch.float32)


class HaltonInit(InitStrategy):
    """Halton sequence (base-2 real, base-3 imag)."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        r = _van_der_corput(n, 2)
        i = _van_der_corput(n, 3)
        wr = ((r * 2 - 1) * scale).view(out_dim, in_dim)
        wi = ((i * 2 - 1) * scale).view(out_dim, in_dim)
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class VanDerCorputInit(InitStrategy):
    """Base-2 bit-reversal sequence."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        v = _van_der_corput(n, 2)
        angles = v * 2 * math.pi
        wr = (scale * torch.cos(angles)).view(out_dim, in_dim)
        wi = (scale * torch.sin(angles)).view(out_dim, in_dim)
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


# ---------------------------------------------------------------------------
# Category 7: Structured linear algebra
# ---------------------------------------------------------------------------

def _hadamard(n: int) -> torch.Tensor:
    """Sylvester construction: H_{2n} = [[H_n, H_n], [H_n, -H_n]]."""
    if n <= 1:
        return torch.ones(1, 1)
    h = _hadamard(n // 2)
    return torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)


class HadamardInit(InitStrategy):
    """+/-1 orthogonal Hadamard matrix."""

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        n = 1 << max(out_dim - 1, in_dim - 1).bit_length()
        n = min(n, 1024)
        h = _hadamard(n)
        wr = h[:out_dim, :in_dim].float() * scale / math.sqrt(n)
        wi = _hadamard(n)[:out_dim, :in_dim].float() * scale / math.sqrt(n)
        if wr.shape[0] < out_dim or wr.shape[1] < in_dim:
            full_wr = torch.randn(out_dim, in_dim, generator=self._gen) * scale * 0.1
            full_wi = torch.randn(out_dim, in_dim, generator=self._gen) * scale * 0.1
            full_wr[: wr.shape[0], : wr.shape[1]] = wr
            full_wi[: wi.shape[0], : wi.shape[1]] = wi
            wr, wi = full_wr, full_wi
        if device:
            wr, wi = wr.to(device), wi.to(device)
        return wr, wi


# ---------------------------------------------------------------------------
# Category 8: SSM-specific
# ---------------------------------------------------------------------------

class HiPPOInit(InitStrategy):
    """HiPPO-LegS from S4: diagonal eigenvalues -(2n+1) mapped to decay bands."""

    def init_ssm_eigenvalues(
        self, state_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = torch.arange(state_dim, dtype=torch.float32)
        eigvals = -(2 * n + 1)
        emin, emax = eigvals.min().item(), eigvals.max().item()
        span = emax - emin + 1e-8
        normalized = (eigvals - emin) / span
        log_A_real = math.log(0.95) + normalized * (math.log(0.999) - math.log(0.95))
        log_A_imag = torch.linspace(0.001, math.pi, state_dim)
        if device:
            log_A_real, log_A_imag = log_A_real.to(device), log_A_imag.to(device)
        return log_A_real, log_A_imag


class S4DLinInit(InitStrategy):
    """S4D diagonal linear: A_n = -1/2 + i*pi*n."""

    def init_ssm_eigenvalues(
        self, state_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = torch.arange(state_dim, dtype=torch.float32)
        log_A_real = torch.full((state_dim,), math.log(0.98))
        log_A_imag = math.pi * n
        if device:
            log_A_real, log_A_imag = log_A_real.to(device), log_A_imag.to(device)
        return log_A_real, log_A_imag


class S4DInvInit(InitStrategy):
    """S4D diagonal inverse: A_n = -1/2 + i*pi*N/(n+1)."""

    def init_ssm_eigenvalues(
        self, state_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = torch.arange(state_dim, dtype=torch.float32)
        log_A_real = torch.full((state_dim,), math.log(0.98))
        log_A_imag = math.pi * state_dim / (n + 1)
        if device:
            log_A_real, log_A_imag = log_A_real.to(device), log_A_imag.to(device)
        return log_A_real, log_A_imag


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

REGISTRY: dict = {
    "random": RandomInit,
    "uniform": UniformInit,
    "orthogonal": OrthogonalInit,
    "golden_ratio": GoldenRatioInit,
    "pi": PiInit,
    "sqrt_primes": SqrtPrimesInit,
    "sinusoidal": SinusoidalInit,
    "circular": CircularInit,
    "roots_of_unity": RootsOfUnityInit,
    "fibonacci_spiral": FibonacciSpiralInit,
    "log_spiral": LogSpiralInit,
    "fermat_spiral": FermatSpiralInit,
    "dft": DFTInit,
    "dct": DCTInit,
    "hartley": HartleyInit,
    "halton": HaltonInit,
    "van_der_corput": VanDerCorputInit,
    "hadamard": HadamardInit,
    "hippo": HiPPOInit,
    "s4d_lin": S4DLinInit,
    "s4d_inv": S4DInvInit,
}


def create_initializer(
    name: str, seed: Optional[int] = None
) -> InitStrategy:
    """Create an initializer by name. Raises ValueError if unknown."""
    name = name.lower().strip()
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown init strategy '{name}'. Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[name](seed=seed)


def list_strategies() -> List[str]:
    """Return sorted list of available strategy names."""
    return sorted(REGISTRY.keys())
