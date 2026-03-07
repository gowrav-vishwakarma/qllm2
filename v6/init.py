"""
Structured initialization strategies for V6 complex-valued architecture.

Forked from V5, extended with:
  - init_ssm_eigenvalues_multiscale(): tiered fast/medium/slow decay lanes
  - init_internal_memory_slots(): phase-spread keys for internal memory
  - init_phase_rotation(): near-identity (unchanged, used by coupler)

V5 benchmarked 13 strategies; orthogonal won (168 PPL vs 349 baseline).
Defaults: --init_strategy orthogonal --init_seed 42.
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
    """Abstract base for initialization strategies."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self._gen = torch.Generator().manual_seed(self.seed)

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Init", "").lower()

    def _scale(self, in_dim: int, out_dim: int) -> float:
        return (2 / (in_dim + out_dim)) ** 0.5

    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        w = torch.randn(out_dim, in_dim, generator=self._gen, device=device) * scale
        return w.clone(), torch.randn(out_dim, in_dim, generator=self._gen, device=device).mul_(scale)

    def init_ssm_eigenvalues(
        self, state_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-timescale: fast/medium/slow lanes instead of uniform linspace."""
        return self.init_ssm_eigenvalues_multiscale(state_dim, device)

    def init_ssm_eigenvalues_multiscale(
        self, state_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tiered decay initialization for multi-timescale SSM."""
        n_fast = int(state_dim * 0.4)
        n_medium = int(state_dim * 0.3)
        n_slow = state_dim - n_fast - n_medium

        fast = torch.linspace(math.log(0.9), math.log(0.99), n_fast)
        medium = torch.linspace(math.log(0.999), math.log(0.9999), n_medium)
        slow = torch.linspace(math.log(0.99999), math.log(0.999999), n_slow)
        log_A_real = torch.cat([fast, medium, slow])

        log_A_imag = torch.linspace(0.001, math.pi, state_dim)
        if device:
            log_A_real, log_A_imag = log_A_real.to(device), log_A_imag.to(device)
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
        r = torch.randn(vocab_size, dim, generator=self._gen, device=device) * std
        i = torch.randn(vocab_size, dim, generator=self._gen, device=device) * std
        return r, i

    def init_skip_connection(
        self, dim: int, std: float = 0.01, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        return torch.randn(dim, 2, generator=self._gen, device=device) * std

    def init_internal_memory_slots(
        self, num_slots: int, dim: int, device: Optional[torch.device] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Initialize internal memory key/value slots.

        Keys are spread across phase space (different angles per slot) for
        retrieval diversity. Values are small random (learned via backprop).

        Returns: ((keys_real, keys_imag), (values_real, values_imag))
        """
        if num_slots == 0:
            empty = torch.zeros(0, dim)
            if device:
                empty = empty.to(device)
            return (empty, empty), (empty, empty)
        angles = torch.linspace(0, 2 * math.pi * (1 - 1 / num_slots), num_slots)
        keys_real = torch.cos(angles).unsqueeze(-1).expand(num_slots, dim).clone() * 0.1
        keys_imag = torch.sin(angles).unsqueeze(-1).expand(num_slots, dim).clone() * 0.1
        keys_real = keys_real + torch.randn(num_slots, dim, generator=self._gen) * 0.01
        keys_imag = keys_imag + torch.randn(num_slots, dim, generator=self._gen) * 0.01

        values_real = torch.randn(num_slots, dim, generator=self._gen) * 0.02
        values_imag = torch.randn(num_slots, dim, generator=self._gen) * 0.02

        if device:
            keys_real, keys_imag = keys_real.to(device), keys_imag.to(device)
            values_real, values_imag = values_real.to(device), values_imag.to(device)
        return (keys_real, keys_imag), (values_real, values_imag)

    def get_info(self) -> dict:
        return {"strategy": self.name, "seed": self.seed}


# ---------------------------------------------------------------------------
# Random variants
# ---------------------------------------------------------------------------

class RandomInit(InitStrategy):
    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        wr = torch.randn(out_dim, in_dim, generator=self._gen, device=device) * scale
        wi = torch.randn(out_dim, in_dim, generator=self._gen, device=device) * scale
        return wr, wi


class UniformInit(InitStrategy):
    def init_complex_linear(
        self, out_dim: int, in_dim: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self._scale(in_dim, out_dim)
        wr = (torch.rand(out_dim, in_dim, generator=self._gen, device=device) * 2 - 1) * scale
        wi = (torch.rand(out_dim, in_dim, generator=self._gen, device=device) * 2 - 1) * scale
        return wr, wi


class OrthogonalInit(InitStrategy):
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
# Number-theoretic
# ---------------------------------------------------------------------------

def _weyl_sequence(n: int, alpha: float) -> torch.Tensor:
    return torch.tensor([(k * alpha) % 1.0 for k in range(n)], dtype=torch.float32)


class PiInit(InitStrategy):
    def init_complex_linear(self, out_dim, in_dim, device=None):
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        angles_r = _weyl_sequence(n, math.pi)
        angles_i = _weyl_sequence(n, math.e)
        mags = (angles_r * 2 - 1) * scale
        wr = (mags * torch.cos(angles_i * 2 * math.pi)).view(out_dim, in_dim)
        wi = (mags * torch.sin(angles_i * 2 * math.pi)).view(out_dim, in_dim)
        if device: wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class SqrtPrimesInit(InitStrategy):
    _primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def init_complex_linear(self, out_dim, in_dim, device=None):
        scale = self._scale(in_dim, out_dim)
        n = out_dim * in_dim
        p = self._primes
        angles_r = _weyl_sequence(n, math.sqrt(p[0]))
        angles_i = _weyl_sequence(n, math.sqrt(p[1]))
        mags = (angles_r * 2 - 1) * scale
        wr = (mags * torch.cos(angles_i * 2 * math.pi)).view(out_dim, in_dim)
        wi = (mags * torch.sin(angles_i * 2 * math.pi)).view(out_dim, in_dim)
        if device: wr, wi = wr.to(device), wi.to(device)
        return wr, wi


# ---------------------------------------------------------------------------
# Trigonometric
# ---------------------------------------------------------------------------

class SinusoidalInit(InitStrategy):
    def init_complex_linear(self, out_dim, in_dim, device=None):
        scale = self._scale(in_dim, out_dim)
        positions = torch.arange(out_dim * in_dim, dtype=torch.float32)
        dim_scale = 10000.0 ** (torch.arange(2, dtype=torch.float32) / 2)
        angles = positions.unsqueeze(-1) / dim_scale
        wr = (scale * torch.sin(angles[:, 0])).view(out_dim, in_dim)
        wi = (scale * torch.cos(angles[:, 1])).view(out_dim, in_dim)
        if device: wr, wi = wr.to(device), wi.to(device)
        return wr, wi


# ---------------------------------------------------------------------------
# Fourier / spectral
# ---------------------------------------------------------------------------

class DFTInit(InitStrategy):
    def init_complex_linear(self, out_dim, in_dim, device=None):
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
            wr = wr + torch.randn(out_dim, in_dim, generator=self._gen) * scale * 0.1
            wi = wi + torch.randn(out_dim, in_dim, generator=self._gen) * scale * 0.1
        wr, wi = wr[:out_dim, :in_dim], wi[:out_dim, :in_dim]
        if device: wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class DCTInit(InitStrategy):
    def init_complex_linear(self, out_dim, in_dim, device=None):
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
            wr, wi = wr + extra, wi + extra
        wr, wi = wr[:out_dim, :in_dim], wi[:out_dim, :in_dim]
        if device: wr, wi = wr.to(device), wi.to(device)
        return wr, wi


class HartleyInit(InitStrategy):
    def init_complex_linear(self, out_dim, in_dim, device=None):
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
            wr, wi = wr + extra, wi + extra
        wr, wi = wr[:out_dim, :in_dim], wi[:out_dim, :in_dim]
        if device: wr, wi = wr.to(device), wi.to(device)
        return wr, wi


# ---------------------------------------------------------------------------
# Structured linear algebra
# ---------------------------------------------------------------------------

def _hadamard(n: int) -> torch.Tensor:
    if n <= 1:
        return torch.ones(1, 1)
    h = _hadamard(n // 2)
    return torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)


class HadamardInit(InitStrategy):
    def init_complex_linear(self, out_dim, in_dim, device=None):
        scale = self._scale(in_dim, out_dim)
        n = 1 << max(out_dim - 1, in_dim - 1).bit_length()
        n = min(n, 1024)
        h = _hadamard(n)
        wr = h[:out_dim, :in_dim].float() * scale / math.sqrt(n)
        wi = _hadamard(n)[:out_dim, :in_dim].float() * scale / math.sqrt(n)
        if wr.shape[0] < out_dim or wr.shape[1] < in_dim:
            full_wr = torch.randn(out_dim, in_dim, generator=self._gen) * scale * 0.1
            full_wi = torch.randn(out_dim, in_dim, generator=self._gen) * scale * 0.1
            full_wr[:wr.shape[0], :wr.shape[1]] = wr
            full_wi[:wi.shape[0], :wi.shape[1]] = wi
            wr, wi = full_wr, full_wi
        if device: wr, wi = wr.to(device), wi.to(device)
        return wr, wi


# ---------------------------------------------------------------------------
# SSM-specific (with multi-timescale adaptations)
# ---------------------------------------------------------------------------

class HiPPOInit(InitStrategy):
    """HiPPO-LegS eigenvalues mapped to multi-timescale tiers."""

    def init_ssm_eigenvalues(self, state_dim, device=None):
        n = torch.arange(state_dim, dtype=torch.float32)
        eigvals = -(2 * n + 1)
        emin, emax = eigvals.min().item(), eigvals.max().item()
        span = emax - emin + 1e-8
        normalized = (eigvals - emin) / span
        # Map HiPPO distribution into multi-timescale range
        log_min = math.log(0.9)      # fast lane floor
        log_max = math.log(0.999999) # slow lane ceiling
        log_A_real = log_min + normalized * (log_max - log_min)
        log_A_imag = torch.linspace(0.001, math.pi, state_dim)
        if device:
            log_A_real, log_A_imag = log_A_real.to(device), log_A_imag.to(device)
        return log_A_real, log_A_imag


class S4DLinInit(InitStrategy):
    """S4D diagonal linear with multi-timescale decay."""

    def init_ssm_eigenvalues(self, state_dim, device=None):
        n = torch.arange(state_dim, dtype=torch.float32)
        n_fast = int(state_dim * 0.4)
        n_medium = int(state_dim * 0.3)
        fast = torch.full((n_fast,), math.log(0.95))
        medium = torch.full((n_medium,), math.log(0.999))
        slow = torch.full((state_dim - n_fast - n_medium,), math.log(0.99999))
        log_A_real = torch.cat([fast, medium, slow])
        log_A_imag = math.pi * n
        if device:
            log_A_real, log_A_imag = log_A_real.to(device), log_A_imag.to(device)
        return log_A_real, log_A_imag


class S4DInvInit(InitStrategy):
    """S4D diagonal inverse with multi-timescale decay."""

    def init_ssm_eigenvalues(self, state_dim, device=None):
        n = torch.arange(state_dim, dtype=torch.float32)
        n_fast = int(state_dim * 0.4)
        n_medium = int(state_dim * 0.3)
        fast = torch.full((n_fast,), math.log(0.95))
        medium = torch.full((n_medium,), math.log(0.999))
        slow = torch.full((state_dim - n_fast - n_medium,), math.log(0.99999))
        log_A_real = torch.cat([fast, medium, slow])
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
    "pi": PiInit,
    "sqrt_primes": SqrtPrimesInit,
    "sinusoidal": SinusoidalInit,
    "dft": DFTInit,
    "dct": DCTInit,
    "hartley": HartleyInit,
    "hadamard": HadamardInit,
    "hippo": HiPPOInit,
    "s4d_lin": S4DLinInit,
    "s4d_inv": S4DInvInit,
}


def create_initializer(
    name: str, seed: Optional[int] = None
) -> InitStrategy:
    name = name.lower().strip()
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown init strategy '{name}'. Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[name](seed=seed)


def list_strategies() -> List[str]:
    return sorted(REGISTRY.keys())
