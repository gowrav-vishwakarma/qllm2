"""
Diffusion primitives for complex-valued V6 architecture.

ComplexNoiseSchedule: cosine/linear noise schedule operating on [dim, 2] tensors.
ComplexTimestepEmbed: sinusoidal timestep -> complex vector via phase encoding.
complex_mse_loss: MSE over both real and imaginary components.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..init import InitStrategy

from .complex import ComplexLinear, ComplexNorm


class ComplexNoiseSchedule:
    """
    Noise schedule for diffusion in complex [dim, 2] space.

    Supports cosine and linear schedules. All noise operations are
    isotropic on real and imaginary components independently.
    """

    def __init__(self, num_steps: int = 1000, schedule: str = 'cosine'):
        self.num_steps = num_steps

        if schedule == 'cosine':
            betas = self._cosine_betas(num_steps)
        elif schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, num_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_betas = betas
        self.register_alphas = alphas
        self.register_alpha_bar = alpha_bar
        self.register_sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.register_sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.register_sqrt_recip_alpha = torch.sqrt(1.0 / alphas)
        self.register_posterior_variance = (
            betas * (1.0 - F.pad(alpha_bar[:-1], (1, 0), value=1.0)) / (1.0 - alpha_bar)
        )

    @staticmethod
    def _cosine_betas(num_steps: int, s: float = 0.008) -> torch.Tensor:
        steps = torch.linspace(0, num_steps, num_steps + 1)
        f_t = torch.cos((steps / num_steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f_t / f_t[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(betas, max=0.999)

    def _get(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Index precomputed schedule values by timestep, reshape for broadcasting."""
        vals = tensor.to(t.device)[t]
        while vals.dim() < 4:
            vals = vals.unsqueeze(-1)
        return vals

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: q(x_t | x_0).

        Args:
            x_0: [B, L, dim, 2] clean complex embeddings
            t: [B] integer timesteps

        Returns:
            x_t: [B, L, dim, 2] noisy embeddings
            noise: [B, L, dim, 2] the noise that was added
        """
        noise = torch.randn_like(x_0)
        sqrt_ab = self._get(self.register_sqrt_alpha_bar, t)
        sqrt_1_ab = self._get(self.register_sqrt_one_minus_alpha_bar, t)
        x_t = sqrt_ab * x_0 + sqrt_1_ab * noise
        return x_t, noise

    def reverse_step(
        self,
        x_t: torch.Tensor,
        predicted_x0: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        DDPM reverse step: p(x_{t-1} | x_t).

        Args:
            x_t: [B, L, dim, 2] current noisy sample
            predicted_x0: [B, L, dim, 2] model's prediction of clean sample
            t: current integer timestep (scalar)
        """
        device = x_t.device
        t_tensor = torch.tensor([t], device=device)

        alpha = self.register_alphas.to(device)[t]
        alpha_bar = self.register_alpha_bar.to(device)[t]
        beta = self.register_betas.to(device)[t]

        x0_coeff = beta * torch.sqrt(
            self.register_alpha_bar.to(device)[max(t - 1, 0)] if t > 0
            else torch.tensor(1.0, device=device)
        ) / (1.0 - alpha_bar)
        xt_coeff = (1.0 - (
            self.register_alpha_bar.to(device)[max(t - 1, 0)] if t > 0
            else torch.tensor(1.0, device=device)
        )) * torch.sqrt(alpha) / (1.0 - alpha_bar)

        mean = x0_coeff * predicted_x0 + xt_coeff * x_t

        if t > 0:
            posterior_var = self.register_posterior_variance.to(device)[t]
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(posterior_var) * noise
        return mean

    def ddim_step(
        self,
        x_t: torch.Tensor,
        predicted_x0: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM step: deterministic (eta=0) or stochastic sampling.

        Args:
            x_t: [B, L, dim, 2] current sample
            predicted_x0: [B, L, dim, 2] predicted clean
            t: current timestep
            t_prev: previous (target) timestep
            eta: stochasticity (0 = deterministic)
        """
        device = x_t.device
        ab = self.register_alpha_bar.to(device)

        alpha_bar_t = ab[t]
        alpha_bar_prev = ab[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        predicted_noise = (x_t - torch.sqrt(alpha_bar_t) * predicted_x0) / (
            torch.sqrt(1.0 - alpha_bar_t) + 1e-8
        )

        sigma = eta * torch.sqrt(
            (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8) *
            (1.0 - alpha_bar_t / alpha_bar_prev)
        )

        direction = torch.sqrt(
            torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)
        ) * predicted_noise

        x_prev = torch.sqrt(alpha_bar_prev) * predicted_x0 + direction
        if eta > 0 and t_prev >= 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)

        return x_prev


class ComplexTimestepEmbed(nn.Module):
    """
    Encode diffusion timestep as a complex [dim, 2] vector via phase encoding.

    Timestep -> sinusoidal features -> ComplexLinear -> [B, 1, dim, 2].
    The phase encoding is natural: different timesteps = different rotations
    in complex space, which the backbone's phase-preserving ops handle natively.
    """

    def __init__(self, dim: int, max_period: int = 10000, initializer=None):
        super().__init__()
        self.dim = dim
        half = dim
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer('freqs', freqs)

        self.proj = ComplexLinear(dim, dim, initializer=initializer)
        self.norm = ComplexNorm(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] integer timesteps
        Returns:
            [B, 1, dim, 2] complex timestep embedding (broadcastable over sequence length)
        """
        t_float = t.float()
        args = t_float.unsqueeze(-1) * self.freqs.unsqueeze(0)
        real = torch.cos(args)
        imag = torch.sin(args)
        z = torch.stack([real, imag], dim=-1)
        z = self.proj(z)
        z = self.norm(z)
        return z.unsqueeze(1)


def complex_mse_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss over complex [dim, 2] tensors.
    Computes mean of (real_diff^2 + imag_diff^2) across all dimensions.
    """
    return ((predicted - target) ** 2).mean()
