"""
V6 Shared Backbone: Banks + Couplers + SSM + Memory.

Extracted from PhaseFieldLM so it can be reused by both
autoregressive (PhaseFieldLM) and diffusion (PhaseFieldDiffusion) models.
The backbone contains ~85%+ of all model parameters.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .config import V6Config
from .core.complex import ComplexLinear, ComplexNorm, cmul, cabs
from .core.ssm import ComplexSSM, SSMState
from .core.bank import NamedBankPair
from .core.coupler import PhaseInterferenceCoupler
from .core.memory import (
    WorkingMemory, InternalMemory, PersistentMemoryReader,
    ExpertMemoryReader,
)
from .core.attention import PhaseAttention


class MemoryFusion(nn.Module):
    """
    Combines outputs from working memory, internal memory, and optionally
    persistent memory using learned complex mixing weights.
    """

    def __init__(
        self,
        dim: int,
        num_sources: int = 2,
        initializer=None,
    ):
        super().__init__()
        self.num_sources = num_sources
        self.mix_projs = nn.ModuleList([
            ComplexLinear(dim, dim, bias=False, initializer=initializer)
            for _ in range(num_sources)
        ])
        self.gate_proj = nn.Linear(dim * num_sources, num_sources)
        self.norm = ComplexNorm(dim)

    def forward(self, *sources: torch.Tensor) -> torch.Tensor:
        mag_feats = [cabs(s) for s in sources]
        gate_in = torch.cat(mag_feats, dim=-1)
        gate_weights = torch.softmax(self.gate_proj(gate_in), dim=-1)

        fused = torch.zeros_like(sources[0])
        for i, (src, proj) in enumerate(zip(sources, self.mix_projs)):
            projected = proj(src)
            w = gate_weights[..., i].unsqueeze(-1).unsqueeze(-1)
            fused = fused + projected * w

        return self.norm(fused)


@dataclass
class BackboneOutput:
    z_out: torch.Tensor
    ssm_state: Optional[SSMState] = None
    wm_keys: Optional[torch.Tensor] = None
    wm_values: Optional[torch.Tensor] = None
    wm_mask: Optional[torch.Tensor] = None
    diversity_loss: Optional[torch.Tensor] = None


class PhaseFieldBackbone(nn.Module):
    """
    Shared V6 processing core: banks + couplers + SSM + memory.

    Used by both PhaseFieldLM (autoregressive) and PhaseFieldDiffusion.
    Accepts an optional timestep_embed for diffusion conditioning.
    """

    def __init__(self, config: V6Config, initializer):
        super().__init__()
        self.config = config

        # Per-layer named banks + coupler
        self.bank_pairs = nn.ModuleList([
            NamedBankPair(
                config.dim, config.bank_expand, config.dropout,
                initializer=initializer,
            )
            for _ in range(config.num_layers)
        ])
        self.couplers = nn.ModuleList([
            PhaseInterferenceCoupler(
                config.dim, num_sources=2, dropout=config.dropout,
                initializer=initializer,
            )
            for _ in range(config.num_layers)
        ])
        self.bank_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0))
            for _ in range(config.num_layers)
        ])

        # Optional PhaseAttention layers (disabled by default)
        self._attn_layer_ids = set()
        self.attn_layers = nn.ModuleDict()
        self.attn_scales = nn.ParameterDict()
        if config.use_attention:
            for i in range(config.num_layers):
                place_here = False
                if config.attn_every > 0 and (i + 1) % config.attn_every == 0:
                    place_here = True
                elif config.attn_every == 0 and i == config.num_layers - 1:
                    place_here = True
                if place_here:
                    self._attn_layer_ids.add(i)
                    self.attn_layers[str(i)] = PhaseAttention(
                        config.dim,
                        num_heads=config.attn_num_heads,
                        window_size=config.attn_window_size,
                        dropout=config.dropout,
                        initializer=initializer,
                    )
                    self.attn_scales[str(i)] = nn.Parameter(torch.tensor(0.1))

        # SSM
        self.ssm = ComplexSSM(
            dim=config.dim,
            state_dim=config.state_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            initializer=initializer,
        )

        # Working memory
        self.working_memory = WorkingMemory(
            config.dim, config.num_wm_slots, config.wm_gate_bias,
            read_topk=config.wm_read_topk,
            slot_decay=config.wm_slot_decay,
            initializer=initializer,
        ) if config.num_wm_slots > 0 else None

        # Internal memory
        self.internal_memory = InternalMemory(
            config.dim, config.num_im_slots,
            read_topk=config.im_read_topk,
            initializer=initializer,
        ) if config.num_im_slots > 0 else None

        # Memory-related modules
        has_memory = config.num_wm_slots > 0 or config.num_im_slots > 0
        has_any_memory = has_memory or config.use_persistent_memory

        self.persistent_reader = (
            PersistentMemoryReader(config.dim, initializer=initializer)
            if config.use_persistent_memory else None
        )
        self.expert_reader = (
            ExpertMemoryReader(config.dim, initializer=initializer)
            if has_memory else None
        )

        if has_memory:
            self.memory_fusion_2 = MemoryFusion(
                config.dim, num_sources=2, initializer=initializer,
            )
            self.memory_fusion_3 = MemoryFusion(
                config.dim, num_sources=3, initializer=initializer,
            )
            self.memory_fusion_4 = MemoryFusion(
                config.dim, num_sources=4, initializer=initializer,
            )
        else:
            self.memory_fusion_2 = None
            self.memory_fusion_3 = None
            self.memory_fusion_4 = None

        self.memory_scale = (
            nn.Parameter(torch.tensor(0.5)) if has_any_memory else None
        )

    def forward(
        self,
        z: torch.Tensor,
        ssm_state: Optional[SSMState] = None,
        wm_keys: Optional[torch.Tensor] = None,
        wm_values: Optional[torch.Tensor] = None,
        wm_mask: Optional[torch.Tensor] = None,
        persistent_keys: Optional[torch.Tensor] = None,
        persistent_values: Optional[torch.Tensor] = None,
        persistent_mask: Optional[torch.Tensor] = None,
        expert_keys: Optional[torch.Tensor] = None,
        expert_values: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
        timestep_embed: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        """
        Process complex tensor through banks + SSM + memory.

        Args:
            z: [B, L, dim, 2] complex input embeddings
            timestep_embed: [B, 1, dim, 2] or None -- added per layer (diffusion only)
        """
        # 1. Banks + Coupler (+ optional Attention) per layer
        bank_z = z
        diversity_losses = []
        for i, (bank_pair, coupler, scale) in enumerate(
            zip(self.bank_pairs, self.couplers, self.bank_scales)
        ):
            residual = bank_z

            if timestep_embed is not None:
                bank_z = bank_z + timestep_embed

            sem_out, ctx_out = bank_pair(bank_z)
            coupled = coupler(sem_out, ctx_out)
            bank_z = residual + coupled * scale

            if i in self._attn_layer_ids:
                attn_out = self.attn_layers[str(i)](bank_z)
                attn_scale = self.attn_scales[str(i)]
                bank_z = bank_z + attn_out * attn_scale

            if self.training:
                dloss = bank_pair.compute_diversity_loss(
                    residual, margin=self.config.diversity_margin,
                )
                diversity_losses.append(dloss)
                dloss_c = coupler.compute_diversity_loss()
                diversity_losses.append(dloss_c)

        # 2. SSM
        ssm_out, new_state = self.ssm(bank_z, ssm_state)

        # 3. Working memory
        new_wm_keys = new_wm_values = new_wm_mask = None
        if self.working_memory is not None:
            wm_retrieved, new_wm_keys, new_wm_values, new_wm_mask = self.working_memory(
                ssm_out, wm_keys, wm_values, wm_mask,
            )

        # 4. Internal memory
        if self.internal_memory is not None:
            im_retrieved = self.internal_memory(ssm_out)

        # 5. Memory fusion
        has_persistent = (persistent_keys is not None and persistent_mask is not None
                          and persistent_mask.sum() > 0)
        has_expert = (expert_keys is not None and expert_mask is not None
                      and expert_mask.sum() > 0)

        memory_sources = []
        if self.working_memory is not None:
            memory_sources.append(wm_retrieved)
        if self.internal_memory is not None:
            memory_sources.append(im_retrieved)

        if has_persistent and self.persistent_reader is not None:
            pm_retrieved = self.persistent_reader(
                ssm_out, persistent_keys, persistent_values, persistent_mask,
            )
            memory_sources.append(pm_retrieved)

        if has_expert and self.expert_reader is not None:
            em_retrieved = self.expert_reader(
                ssm_out, expert_keys, expert_values, expert_mask,
            )
            memory_sources.append(em_retrieved)

        n_sources = len(memory_sources)
        if n_sources == 0:
            z_out = ssm_out
        elif n_sources == 1:
            z_out = ssm_out + memory_sources[0] * self.memory_scale
        elif n_sources == 2 and self.memory_fusion_2 is not None:
            memory_out = self.memory_fusion_2(*memory_sources)
            z_out = ssm_out + memory_out * self.memory_scale
        elif n_sources == 3 and self.memory_fusion_3 is not None:
            memory_out = self.memory_fusion_3(*memory_sources)
            z_out = ssm_out + memory_out * self.memory_scale
        elif n_sources >= 4 and self.memory_fusion_4 is not None:
            memory_out = self.memory_fusion_4(*memory_sources)
            z_out = ssm_out + memory_out * self.memory_scale
        else:
            z_out = ssm_out

        # Diversity loss
        div_loss = None
        if diversity_losses:
            div_loss = torch.stack(diversity_losses).mean()

        return BackboneOutput(
            z_out=z_out,
            ssm_state=new_state,
            wm_keys=new_wm_keys,
            wm_values=new_wm_values,
            wm_mask=new_wm_mask,
            diversity_loss=div_loss,
        )

    def count_parameters(self) -> Dict[str, int]:
        counts = {
            'banks': sum(p.numel() for p in self.bank_pairs.parameters()),
            'couplers': sum(p.numel() for p in self.couplers.parameters()),
            'attention': (
                sum(p.numel() for p in self.attn_layers.parameters()) +
                sum(p.numel() for p in self.attn_scales.parameters())
            ),
            'ssm': sum(p.numel() for p in self.ssm.parameters()),
            'working_memory': sum(p.numel() for p in self.working_memory.parameters()) if self.working_memory else 0,
            'internal_memory': sum(p.numel() for p in self.internal_memory.parameters()) if self.internal_memory else 0,
            'persistent_reader': sum(p.numel() for p in self.persistent_reader.parameters()) if self.persistent_reader else 0,
            'expert_reader': sum(p.numel() for p in self.expert_reader.parameters()) if self.expert_reader else 0,
            'memory_fusion': (
                (sum(p.numel() for p in self.memory_fusion_2.parameters()) if self.memory_fusion_2 else 0) +
                (sum(p.numel() for p in self.memory_fusion_3.parameters()) if self.memory_fusion_3 else 0) +
                (sum(p.numel() for p in self.memory_fusion_4.parameters()) if self.memory_fusion_4 else 0)
            ),
            'memory_scale': 1 if self.memory_scale is not None else 0,
        }
        return counts
