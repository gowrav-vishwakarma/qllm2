#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Byte Patching: Fixed-size byte grouping for efficient byte-level LM

Core concept:
- Group raw bytes into fixed-size patches (e.g., P=4 bytes per patch)
- Process patch latents through backbone (reduces seq length by P)
- Decode patch outputs back to per-byte logits for byte-level CE

This allows byte-level objectives with vastly reduced compute cost.

Architecture:
    ByteIDs[T] -> BytePatcher -> PatchLatents[L=T/P, dim, 2]
                                   |
                                   v
                              [Backbone processing]
                                   |
                                   v
    PatchStates[L, dim, 2] -> WithinPatchByteDecoder -> ByteLogits[T, 259]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .phase2d import (
    Phase2DEmbed, Phase2DLinear, Phase2DLayerNorm,
    phase2d_multiply, phase2d_normalize, phase2d_to_real,
)


@dataclass
class BytePatchInfo:
    """Metadata for byte patching (used to reconstruct byte logits)"""
    patch_size: int
    num_patches: int  # L
    original_len: int  # T (before padding)
    padded_len: int  # T_padded (multiple of patch_size)
    pad_len: int  # Number of padding bytes added


class BytePatcher(nn.Module):
    """
    Fixed-size byte patcher: converts byte IDs to patch latents.
    
    Input: [B, T] byte IDs (0-258: bytes + BOS + EOS + PAD)
    Output: [B, L, dim, 2] patch latents where L = ceil(T / P)
    
    Uses learnable position weights to combine bytes within each patch.
    """
    
    def __init__(
        self,
        vocab_size: int = 259,  # 256 bytes + PAD(256) + BOS(257) + EOS(258)
        dim: int = 256,
        patch_size: int = 4,
        padding_idx: int = 256,  # PAD token
    ):
        """
        Args:
            vocab_size: Byte vocabulary size (typically 259)
            dim: Phase dimension
            patch_size: Number of bytes per patch (P)
            padding_idx: Padding token ID
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.patch_size = patch_size
        self.padding_idx = padding_idx
        
        # Byte embedding (Phase2D)
        self.byte_embed = Phase2DEmbed(
            vocab_size=vocab_size,
            dim=dim,
            padding_idx=padding_idx,
        )
        
        # Learnable position weights for aggregating bytes within patch
        # Shape: [P] - weight for each position in patch
        self.position_weights = nn.Parameter(torch.ones(patch_size) / patch_size)
        
        # Per-position projection before aggregation
        self.position_proj = nn.ModuleList([
            Phase2DLinear(dim, dim) for _ in range(patch_size)
        ])
        
        # Output projection after aggregation
        self.output_proj = Phase2DLinear(dim, dim)
        self.output_norm = Phase2DLayerNorm(dim)
    
    def pad_to_patch_multiple(
        self,
        input_ids: torch.Tensor,  # [B, T]
    ) -> Tuple[torch.Tensor, BytePatchInfo]:
        """
        Pad input to be a multiple of patch_size.
        
        Returns:
            padded_ids: [B, T_padded] where T_padded is multiple of patch_size
            info: BytePatchInfo with metadata
        """
        batch_size, orig_len = input_ids.shape
        
        # Calculate padding needed
        remainder = orig_len % self.patch_size
        pad_len = (self.patch_size - remainder) % self.patch_size
        padded_len = orig_len + pad_len
        num_patches = padded_len // self.patch_size
        
        # Pad if needed
        if pad_len > 0:
            pad_tokens = torch.full(
                (batch_size, pad_len),
                self.padding_idx,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            padded_ids = torch.cat([input_ids, pad_tokens], dim=1)
        else:
            padded_ids = input_ids
        
        info = BytePatchInfo(
            patch_size=self.patch_size,
            num_patches=num_patches,
            original_len=orig_len,
            padded_len=padded_len,
            pad_len=pad_len,
        )
        
        return padded_ids, info
    
    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
    ) -> Tuple[torch.Tensor, BytePatchInfo]:
        """
        Convert byte IDs to patch latents.
        
        Args:
            input_ids: [B, T] byte IDs
        
        Returns:
            patch_latents: [B, L, dim, 2] Phase2D patch representations
            info: BytePatchInfo metadata for decoding
        """
        # 1. Pad to patch multiple
        padded_ids, info = self.pad_to_patch_multiple(input_ids)
        batch_size = padded_ids.shape[0]
        
        # 2. Embed bytes
        byte_embeds = self.byte_embed(padded_ids)  # [B, T_padded, dim, 2]
        
        # 3. Reshape to [B, L, P, dim, 2]
        byte_embeds = byte_embeds.view(
            batch_size,
            info.num_patches,
            self.patch_size,
            self.dim,
            2,
        )
        
        # 4. Apply per-position projection
        projected = []
        for p in range(self.patch_size):
            pos_embed = byte_embeds[:, :, p, :, :]  # [B, L, dim, 2]
            pos_proj = self.position_proj[p](pos_embed)  # [B, L, dim, 2]
            projected.append(pos_proj)
        
        projected = torch.stack(projected, dim=2)  # [B, L, P, dim, 2]
        
        # 5. Aggregate with position weights (normalized)
        weights = F.softmax(self.position_weights, dim=0)  # [P]
        weights = weights.view(1, 1, self.patch_size, 1, 1)  # [1, 1, P, 1, 1]
        
        patch_latents = (projected * weights).sum(dim=2)  # [B, L, dim, 2]
        
        # 6. Output projection and normalization
        patch_latents = self.output_proj(patch_latents)
        patch_latents = self.output_norm(patch_latents)
        
        return patch_latents, info


class WithinPatchByteDecoder(nn.Module):
    """
    Within-patch byte decoder: converts patch latents back to per-byte logits.
    
    Uses teacher forcing during training: previous bytes are embedded and
    used to condition each position's prediction within the patch.
    
    Input: [B, L, dim, 2] patch states + [B, T] input_ids (for teacher forcing)
    Output: [B, T, vocab_size] byte logits
    """
    
    def __init__(
        self,
        vocab_size: int = 259,
        dim: int = 256,
        patch_size: int = 4,
        num_layers: int = 2,
        padding_idx: int = 256,
    ):
        """
        Args:
            vocab_size: Byte vocabulary size
            dim: Phase dimension
            patch_size: Number of bytes per patch
            num_layers: Number of decoder layers
            padding_idx: Padding token ID
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        
        # Byte embedding for teacher-forced previous bytes
        self.byte_embed = Phase2DEmbed(
            vocab_size=vocab_size,
            dim=dim,
            padding_idx=padding_idx,
        )
        
        # Position embedding within patch (0 to P-1)
        self.position_embed = nn.Parameter(torch.randn(patch_size, dim, 2) * 0.02)
        
        # Causal decoder layers (lightweight: P is small, typically 4)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'proj': Phase2DLinear(dim, dim),
                'norm': Phase2DLayerNorm(dim),
            }))
        
        # Output head: Phase2D -> logits
        self.output_proj = Phase2DLinear(dim, dim)
        self.output_norm = Phase2DLayerNorm(dim)
        self.lm_head = nn.Linear(dim * 2, vocab_size, bias=False)
        
        # Causal mask for within-patch attention (cached)
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(patch_size, patch_size)).bool(),
            persistent=False,
        )
    
    def _create_shifted_input(
        self,
        input_ids: torch.Tensor,  # [B, T]
        info: BytePatchInfo,
    ) -> torch.Tensor:
        """
        Create shifted input for teacher forcing.
        
        For each position p in patch, use byte at position p-1 as input.
        Position 0 uses a special start token (we use PAD as placeholder).
        
        Returns:
            [B, L, P] shifted byte IDs
        """
        batch_size = input_ids.shape[0]
        
        # Pad input to patch multiple if needed
        if input_ids.shape[1] < info.padded_len:
            pad_tokens = torch.full(
                (batch_size, info.padded_len - input_ids.shape[1]),
                self.padding_idx,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            padded_ids = torch.cat([input_ids, pad_tokens], dim=1)
        else:
            padded_ids = input_ids[:, :info.padded_len]
        
        # Shift: prepend a start token and remove last token
        start_token = torch.full(
            (batch_size, 1),
            self.padding_idx,  # Use PAD as start token
            dtype=padded_ids.dtype,
            device=padded_ids.device,
        )
        shifted = torch.cat([start_token, padded_ids[:, :-1]], dim=1)
        
        # Reshape to [B, L, P]
        shifted = shifted.view(batch_size, info.num_patches, self.patch_size)
        
        return shifted
    
    def forward(
        self,
        patch_states: torch.Tensor,  # [B, L, dim, 2]
        input_ids: torch.Tensor,  # [B, T] original input for teacher forcing
        info: BytePatchInfo,
    ) -> torch.Tensor:
        """
        Decode patch states to per-byte logits.
        
        Args:
            patch_states: [B, L, dim, 2] output from backbone
            input_ids: [B, T] original byte IDs (for teacher forcing)
            info: BytePatchInfo from patcher
        
        Returns:
            logits: [B, T, vocab_size] per-byte logits (trimmed to original length)
        """
        batch_size, num_patches, dim, _ = patch_states.shape
        
        # 1. Create shifted input for teacher forcing
        shifted_ids = self._create_shifted_input(input_ids, info)  # [B, L, P]
        
        # 2. Embed shifted bytes
        # Reshape to [B*L, P] for embedding
        shifted_flat = shifted_ids.view(batch_size * num_patches, self.patch_size)
        prev_byte_embed = self.byte_embed(shifted_flat)  # [B*L, P, dim, 2]
        prev_byte_embed = prev_byte_embed.view(
            batch_size, num_patches, self.patch_size, dim, 2
        )  # [B, L, P, dim, 2]
        
        # 3. Add position embedding within patch
        pos_embed = self.position_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, P, dim, 2]
        prev_byte_embed = prev_byte_embed + pos_embed
        
        # 4. Add patch state (broadcast to all positions in patch)
        patch_broadcast = patch_states.unsqueeze(2)  # [B, L, 1, dim, 2]
        x = prev_byte_embed + patch_broadcast  # [B, L, P, dim, 2]
        
        # 5. Apply causal decoder layers
        # Reshape to [B*L, P, dim, 2] for processing
        x = x.view(batch_size * num_patches, self.patch_size, dim, 2)
        
        for layer in self.layers:
            # Simple causal processing via masked addition
            # For small P (e.g., 4), we can do a causal scan
            residual = x
            
            # Causal aggregation: each position sees previous positions
            # Using cumulative sum with masking
            x_cumsum = torch.cumsum(x, dim=1)  # [B*L, P, dim, 2]
            positions = torch.arange(
                1, self.patch_size + 1, device=x.device, dtype=x.dtype
            ).view(1, self.patch_size, 1, 1)
            x_mean = x_cumsum / positions  # Causal mean
            
            x = layer['proj'](x_mean)
            x = layer['norm'](x)
            x = x + residual
        
        # 6. Output projection
        x = self.output_proj(x)
        x = self.output_norm(x)
        
        # 7. Convert Phase2D to real and project to logits
        # x shape: [B*L, P, dim, 2]
        x_real = phase2d_to_real(x, mode='concat')  # [B*L, P, dim*2]
        logits = self.lm_head(x_real)  # [B*L, P, vocab_size]
        
        # 8. Reshape back to [B, L*P, vocab_size] and trim to original length
        logits = logits.view(batch_size, num_patches * self.patch_size, self.vocab_size)
        logits = logits[:, :info.original_len, :]  # [B, T, vocab_size]
        
        return logits
    
    def generate_patch(
        self,
        patch_state: torch.Tensor,  # [B, dim, 2] single patch state
        prev_byte: Optional[torch.Tensor] = None,  # [B] previous byte (from prior patch)
    ) -> torch.Tensor:
        """
        Generate bytes for a single patch autoregressively.
        
        Used during inference to decode patch-by-patch.
        
        Args:
            patch_state: [B, dim, 2] single patch state from backbone
            prev_byte: [B] last byte from previous patch (or PAD for first)
        
        Returns:
            [B, P] generated byte IDs for this patch
        """
        batch_size = patch_state.shape[0]
        device = patch_state.device
        
        # Initialize with previous byte or PAD
        if prev_byte is None:
            prev_byte = torch.full(
                (batch_size,),
                self.padding_idx,
                dtype=torch.long,
                device=device,
            )
        
        generated = []
        current_byte = prev_byte
        
        for p in range(self.patch_size):
            # Embed previous byte
            byte_embed = self.byte_embed(current_byte.unsqueeze(1))  # [B, 1, dim, 2]
            byte_embed = byte_embed.squeeze(1)  # [B, dim, 2]
            
            # Add position embedding
            pos_embed = self.position_embed[p]  # [dim, 2]
            byte_embed = byte_embed + pos_embed
            
            # Combine with patch state
            x = byte_embed + patch_state  # [B, dim, 2]
            
            # Apply layers (simplified for single position)
            for layer in self.layers:
                residual = x
                x = layer['proj'](x)
                x = layer['norm'](x)
                x = x + residual
            
            # Output projection
            x = self.output_proj(x)
            x = self.output_norm(x)
            
            # Get logits
            x_real = phase2d_to_real(x, mode='concat')  # [B, dim*2]
            logits = self.lm_head(x_real)  # [B, vocab_size]
            
            # Sample next byte (greedy for now)
            next_byte = logits.argmax(dim=-1)  # [B]
            generated.append(next_byte)
            current_byte = next_byte
        
        return torch.stack(generated, dim=1)  # [B, P]


class BytePatchingModule(nn.Module):
    """
    Combined byte patching module for easy integration into model.
    
    Wraps BytePatcher + WithinPatchByteDecoder with shared vocabulary.
    """
    
    def __init__(
        self,
        vocab_size: int = 259,
        dim: int = 256,
        patch_size: int = 4,
        decoder_layers: int = 2,
        padding_idx: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.patch_size = patch_size
        
        self.patcher = BytePatcher(
            vocab_size=vocab_size,
            dim=dim,
            patch_size=patch_size,
            padding_idx=padding_idx,
        )
        
        self.decoder = WithinPatchByteDecoder(
            vocab_size=vocab_size,
            dim=dim,
            patch_size=patch_size,
            num_layers=decoder_layers,
            padding_idx=padding_idx,
        )
    
    def encode(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, BytePatchInfo]:
        """Encode bytes to patch latents."""
        return self.patcher(input_ids)
    
    def decode(
        self,
        patch_states: torch.Tensor,
        input_ids: torch.Tensor,
        info: BytePatchInfo,
    ) -> torch.Tensor:
        """Decode patch states to byte logits."""
        return self.decoder(patch_states, input_ids, info)
