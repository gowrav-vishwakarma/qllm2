"""
V6 Phase-Domain Diffusion Model.

Shares the PhaseFieldBackbone with PhaseFieldLM (autoregressive).
Adds timestep conditioning and mode-specific encoder/decoder for
text diffusion and image diffusion.

Usage:
    from v6.model import create_model
    config.mode = 'diffusion_text'   # or 'diffusion_image'
    model = create_model(config)     # returns PhaseFieldDiffusion
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass

from .config import V6Config
from .init import create_initializer
from .core.complex import (
    ComplexEmbed, ComplexLinear, ComplexNorm, cabs,
)
from .core.diffusion import (
    ComplexNoiseSchedule, ComplexTimestepEmbed, complex_mse_loss,
)
from .backbone import PhaseFieldBackbone


@dataclass
class DiffusionOutput:
    loss: torch.Tensor
    predicted: torch.Tensor
    diversity_loss: Optional[torch.Tensor] = None


class PhaseFieldDiffusion(nn.Module):
    """
    Diffusion model using the shared V6 backbone.

    Handles both text and image modalities via config.mode:
    - 'diffusion_text': ComplexEmbed encoder, project-to-embedding decoder
    - 'diffusion_image': image encoder/decoder (patch or FFT, loaded lazily)
    """

    def __init__(self, config: V6Config):
        super().__init__()
        self.config = config

        initializer = create_initializer(config.init_strategy, config.init_seed)
        config.init_seed = initializer.seed

        # Shared backbone
        self.backbone = PhaseFieldBackbone(config, initializer)

        # Timestep conditioning
        self.time_embed = ComplexTimestepEmbed(config.dim, initializer=initializer)

        # Noise schedule (not an nn.Module, just precomputed tensors)
        self.noise_schedule = ComplexNoiseSchedule(
            config.diffusion_steps, config.noise_schedule,
        )

        # Mode-specific encoder/decoder
        # Output projection always operates in latent complex space [B, L, dim, 2].
        # For images, the pixel-space decoder is only used during sampling.
        self.output_proj = ComplexLinear(
            config.dim, config.dim, initializer=initializer,
        )
        self.output_norm = ComplexNorm(config.dim)

        if config.mode == 'diffusion_text':
            self.encoder = ComplexEmbed(
                config.vocab_size, config.dim, initializer=initializer,
            )
            self.encoder_norm = ComplexNorm(config.dim)
        elif config.mode == 'diffusion_image':
            from .core.image_codec import create_image_encoder, create_image_decoder
            self.encoder = create_image_encoder(config, initializer)
            self.decoder = create_image_decoder(config, initializer)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to complex space [B, L, dim, 2]."""
        z = self.encoder(x)
        if hasattr(self, 'encoder_norm'):
            z = self.encoder_norm(z)
        return z

    def _decode(self, z_out: torch.Tensor) -> torch.Tensor:
        """Project backbone output in latent complex space (for loss computation)."""
        return self.output_norm(self.output_proj(z_out))

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> DiffusionOutput:
        """
        Training forward pass.

        Args:
            x: input data
               - text mode: token IDs [B, L]
               - image mode: images [B, C, H, W]
            t: timesteps [B], sampled randomly if None

        Returns:
            DiffusionOutput with loss, predicted output, and diversity loss
        """
        z_0 = self._encode(x)
        B = z_0.shape[0]

        if t is None:
            t = torch.randint(0, self.config.diffusion_steps, (B,), device=z_0.device)

        z_t, noise = self.noise_schedule.add_noise(z_0, t)
        t_emb = self.time_embed(t)

        bb = self.backbone(z_t, timestep_embed=t_emb)
        predicted = self._decode(bb.z_out)

        if self.config.prediction_target == 'x0':
            target = z_0
        elif self.config.prediction_target == 'epsilon':
            target = noise
        else:
            raise ValueError(f"Unknown prediction target: {self.config.prediction_target}")

        if self.config.diffusion_loss == 'mse':
            loss = complex_mse_loss(predicted, target)
        elif self.config.diffusion_loss == 'huber':
            loss = nn.functional.smooth_l1_loss(predicted, target)
        else:
            raise ValueError(f"Unknown loss: {self.config.diffusion_loss}")

        return DiffusionOutput(
            loss=loss,
            predicted=predicted,
            diversity_loss=bb.diversity_loss,
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples via iterative denoising.

        Args:
            batch_size: number of samples to generate
            seq_len: sequence length (num tokens for text, num patches/positions for image)
            device: target device
            num_steps: override diffusion steps (defaults to config)

        Returns:
            For text: token IDs [B, L]
            For image: pixel tensor [B, C, H, W]
        """
        self.eval()
        steps = num_steps or self.config.diffusion_steps

        z_t = torch.randn(batch_size, seq_len, self.config.dim, 2, device=device)

        if self.config.sampling_method == 'ddpm':
            for t_val in reversed(range(steps)):
                t_emb = self.time_embed(
                    torch.tensor([t_val], device=device).expand(batch_size)
                )
                bb = self.backbone(z_t, timestep_embed=t_emb)
                predicted = self._decode(bb.z_out)

                if self.config.prediction_target == 'x0':
                    predicted_x0 = predicted
                else:
                    # Convert epsilon prediction to x0
                    ab = self.noise_schedule.register_alpha_bar.to(device)[t_val]
                    predicted_x0 = (z_t - torch.sqrt(1 - ab) * predicted) / (torch.sqrt(ab) + 1e-8)

                z_t = self.noise_schedule.reverse_step(z_t, predicted_x0, t_val)

        elif self.config.sampling_method == 'ddim':
            ddim_steps = self.config.ddim_steps
            step_size = steps // ddim_steps
            timesteps = list(range(0, steps, step_size))[::-1]

            for i, t_val in enumerate(timesteps):
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                t_emb = self.time_embed(
                    torch.tensor([t_val], device=device).expand(batch_size)
                )
                bb = self.backbone(z_t, timestep_embed=t_emb)
                predicted = self._decode(bb.z_out)

                if self.config.prediction_target == 'x0':
                    predicted_x0 = predicted
                else:
                    ab = self.noise_schedule.register_alpha_bar.to(device)[t_val]
                    predicted_x0 = (z_t - torch.sqrt(1 - ab) * predicted) / (torch.sqrt(ab) + 1e-8)

                z_t = self.noise_schedule.ddim_step(
                    z_t, predicted_x0, t_val, t_prev, eta=self.config.ddim_eta,
                )

        return self._decode_to_output(z_t)

    def _decode_to_output(self, z: torch.Tensor) -> torch.Tensor:
        """Convert denoised complex embeddings to final output format."""
        if self.config.mode == 'diffusion_text':
            logits = (
                z[..., 0] @ self.encoder.embed_real.weight.T +
                z[..., 1] @ self.encoder.embed_imag.weight.T
            )
            return logits.argmax(dim=-1)
        elif self.config.mode == 'diffusion_image':
            return self.decoder(z)
        else:
            return z

    @property
    def initializer_info(self) -> dict:
        return {
            "init_strategy": self.config.init_strategy,
            "init_seed": self.config.init_seed,
        }

    def count_parameters(self) -> Dict[str, int]:
        bb_counts = self.backbone.count_parameters()
        counts = {
            'time_embed': sum(p.numel() for p in self.time_embed.parameters()),
            **bb_counts,
            'output_proj': (
                sum(p.numel() for p in self.output_proj.parameters()) +
                sum(p.numel() for p in self.output_norm.parameters())
            ),
        }
        if self.config.mode == 'diffusion_text':
            counts['encoder (embed)'] = sum(p.numel() for p in self.encoder.parameters())
        elif self.config.mode == 'diffusion_image':
            counts['image_encoder'] = sum(p.numel() for p in self.encoder.parameters())
            counts['image_decoder'] = sum(p.numel() for p in self.decoder.parameters())
        counts['total'] = sum(counts.values())
        return counts
