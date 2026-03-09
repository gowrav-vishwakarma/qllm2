"""
Image encoders and decoders for V6 diffusion.

Two approaches:
  - Patch: ViT-style split into patches, project each to complex space.
    Simple, proven, good for quick prototyping.
  - FFT: 2D FFT per channel to get complex coefficients, then project
    to model dimension. Leverages V6's native complex representation and
    the SSM's multi-timescale properties (low freq -> slow lanes, high freq -> fast).

Both produce [B, num_positions, dim, 2] complex tensors for the backbone.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex import ComplexLinear, ComplexNorm


# ---------------------------------------------------------------------------
# Patch Encoder / Decoder
# ---------------------------------------------------------------------------

class PatchImageEncoder(nn.Module):
    """
    Split image into non-overlapping patches and project to complex space.
    image_size=64, patch_size=8 -> 64 patches, each 8*8*3=192 values.
    """

    def __init__(self, image_size, patch_size, channels, dim, initializer=None):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * channels
        self.proj = ComplexLinear(patch_dim, dim, initializer=initializer)
        self.norm = ComplexNorm(dim)
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] images in [-1, 1]
        Returns:
            [B, num_patches, dim, 2] complex embeddings
        """
        B = x.shape[0]
        patches = self.unfold(x)  # [B, patch_dim, num_patches]
        patches = patches.transpose(1, 2)  # [B, num_patches, patch_dim]

        z_real = patches
        z_imag = torch.zeros_like(patches)
        z = torch.stack([z_real, z_imag], dim=-1)  # [B, num_patches, patch_dim, 2]

        z = self.proj(z)
        z = self.norm(z)
        return z


class PatchImageDecoder(nn.Module):
    """Project complex space back to patches and fold into an image."""

    def __init__(self, image_size, patch_size, channels, dim, initializer=None):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        patch_dim = patch_size * patch_size * channels
        self.proj = ComplexLinear(dim, patch_dim, initializer=initializer)
        self.fold = nn.Fold(
            output_size=(image_size, image_size),
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, num_patches, dim, 2] complex embeddings
        Returns:
            [B, C, H, W] reconstructed image
        """
        z = self.proj(z)  # [B, num_patches, patch_dim, 2]
        patches = z[..., 0]  # take real part as pixel values

        patches = patches.transpose(1, 2)  # [B, patch_dim, num_patches]
        img = self.fold(patches)  # [B, C, H, W]
        return img.clamp(-1, 1)


# ---------------------------------------------------------------------------
# FFT Encoder / Decoder
# ---------------------------------------------------------------------------

class FFTImageEncoder(nn.Module):
    """
    2D FFT per channel -> complex coefficients -> project to model dim.

    This approach is natural for V6: the FFT gives us complex-valued
    frequency representations, and the SSM's multi-timescale decay lanes
    can separate low-frequency structure (slow lanes) from high-frequency
    detail (fast lanes).
    """

    def __init__(self, image_size, channels, dim, initializer=None):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.proj = ComplexLinear(channels, dim, initializer=initializer)
        self.norm = ComplexNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] images in [-1, 1]
        Returns:
            [B, H*W, dim, 2] complex embeddings
        """
        B, C, H, W = x.shape

        freq = torch.fft.fft2(x, norm='ortho')  # [B, C, H, W] complex
        freq_real = freq.real  # [B, C, H, W]
        freq_imag = freq.imag  # [B, C, H, W]

        # Reshape: [B, C, H, W] -> [B, H*W, C]
        freq_real = freq_real.permute(0, 2, 3, 1).reshape(B, H * W, C)
        freq_imag = freq_imag.permute(0, 2, 3, 1).reshape(B, H * W, C)

        z = torch.stack([freq_real, freq_imag], dim=-1)  # [B, H*W, C, 2]

        z = self.proj(z)  # [B, H*W, dim, 2]
        z = self.norm(z)
        return z


class FFTImageDecoder(nn.Module):
    """Project back to frequency domain, inverse FFT to pixel space."""

    def __init__(self, image_size, channels, dim, initializer=None):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.proj = ComplexLinear(dim, channels, initializer=initializer)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, H*W, dim, 2] complex embeddings
        Returns:
            [B, C, H, W] reconstructed image
        """
        B = z.shape[0]
        H = W = self.image_size

        z = self.proj(z)  # [B, H*W, C, 2]
        freq_real = z[..., 0]  # [B, H*W, C]
        freq_imag = z[..., 1]  # [B, H*W, C]

        # Reshape: [B, H*W, C] -> [B, C, H, W]
        freq_real = freq_real.reshape(B, H, W, self.channels).permute(0, 3, 1, 2)
        freq_imag = freq_imag.reshape(B, H, W, self.channels).permute(0, 3, 1, 2)

        freq = torch.complex(freq_real, freq_imag)
        img = torch.fft.ifft2(freq, norm='ortho').real  # [B, C, H, W]
        return img.clamp(-1, 1)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_image_encoder(config, initializer=None):
    if config.image_encoder == 'patch':
        return PatchImageEncoder(
            config.image_size, config.patch_size, config.image_channels,
            config.dim, initializer=initializer,
        )
    elif config.image_encoder == 'fft':
        return FFTImageEncoder(
            config.image_size, config.image_channels,
            config.dim, initializer=initializer,
        )
    else:
        raise ValueError(f"Unknown image encoder: {config.image_encoder}")


def create_image_decoder(config, initializer=None):
    if config.image_encoder == 'patch':
        return PatchImageDecoder(
            config.image_size, config.patch_size, config.image_channels,
            config.dim, initializer=initializer,
        )
    elif config.image_encoder == 'fft':
        return FFTImageDecoder(
            config.image_size, config.image_channels,
            config.dim, initializer=initializer,
        )
    else:
        raise ValueError(f"Unknown image encoder: {config.image_encoder}")
