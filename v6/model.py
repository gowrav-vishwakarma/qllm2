"""
V6 Phase-First Language Model.

Architecture (no attention anywhere):
    Tokens -> ComplexEmbed
    -> [NamedBankPair -> PhaseInterferenceCoupler] x N layers
    -> MultiTimescaleSSM
    -> WorkingMemory (learned write/read, per-sequence)
    -> InternalMemory (trained slots, general knowledge)
    -> [PersistentMemory (external, per-user)] (optional)
    -> MemoryFusion (learned complex mixing)
    -> TiedComplexLMHead
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .config import V6Config
from .init import create_initializer
from .core.complex import (
    ComplexEmbed, ComplexLinear, ComplexNorm, cmul, cabs,
)
from .core.ssm import ComplexSSM, SSMState
from .core.bank import NamedBankPair
from .core.coupler import PhaseInterferenceCoupler
from .core.memory import (
    WorkingMemory, InternalMemory, PersistentMemoryReader,
    ExpertMemoryReader,
)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    ssm_state: Optional[SSMState] = None
    wm_keys: Optional[torch.Tensor] = None
    wm_values: Optional[torch.Tensor] = None
    wm_mask: Optional[torch.Tensor] = None
    diversity_loss: Optional[torch.Tensor] = None


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
        """
        sources: tuple of [B, L, dim, 2] tensors from different memory types.
        Returns: [B, L, dim, 2] fused output.
        """
        # Gate weights from magnitude features
        mag_features = [cabs(s).mean(dim=-1) for s in sources]  # list of [B, L]
        # Expand to have dim features for gating
        mag_feats = [cabs(s) for s in sources]  # list of [B, L, dim]
        gate_in = torch.cat(mag_feats, dim=-1)  # [B, L, dim*num_sources]
        gate_weights = torch.softmax(self.gate_proj(gate_in), dim=-1)  # [B, L, num_sources]

        fused = torch.zeros_like(sources[0])
        for i, (src, proj) in enumerate(zip(sources, self.mix_projs)):
            projected = proj(src)
            w = gate_weights[..., i].unsqueeze(-1).unsqueeze(-1)
            fused = fused + projected * w

        return self.norm(fused)


class PhaseFieldLM(nn.Module):
    """
    V6 Phase-First Language Model.

    No attention anywhere. Long-context coherence via:
    1. Multi-timescale SSM (fast/medium/slow decay lanes)
    2. Working memory (learned write/read, non-decaying per-sequence)
    3. Internal memory (trained slots, general knowledge)
    4. Optional persistent memory (per-user, loaded from disk)
    """

    def __init__(self, config: V6Config):
        super().__init__()
        self.config = config

        initializer = create_initializer(config.init_strategy, config.init_seed)
        config.init_seed = initializer.seed

        # Embedding
        self.embed = ComplexEmbed(
            config.vocab_size, config.dim, initializer=initializer
        )
        self.embed_norm = ComplexNorm(config.dim)

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

        # SSM backbone
        self.ssm = ComplexSSM(
            dim=config.dim,
            state_dim=config.state_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            initializer=initializer,
        )

        # Working memory (None if disabled via --no_working_memory)
        self.working_memory = WorkingMemory(
            config.dim, config.num_wm_slots, config.wm_gate_bias,
            initializer=initializer,
        ) if config.num_wm_slots > 0 else None

        # Internal memory (None if disabled via --no_internal_memory)
        self.internal_memory = InternalMemory(
            config.dim, config.num_im_slots,
            initializer=initializer,
        ) if config.num_im_slots > 0 else None

        # Persistent memory reader (projections only; actual memory is external)
        self.persistent_reader = PersistentMemoryReader(
            config.dim, initializer=initializer,
        )

        # Expert memory reader (projections only; actual memory is external)
        self.expert_reader = ExpertMemoryReader(
            config.dim, initializer=initializer,
        )

        # Memory fusion: working + internal (+ persistent/expert when available)
        self.memory_fusion_2 = MemoryFusion(
            config.dim, num_sources=2, initializer=initializer,
        )
        self.memory_fusion_3 = MemoryFusion(
            config.dim, num_sources=3, initializer=initializer,
        )
        self.memory_fusion_4 = MemoryFusion(
            config.dim, num_sources=4, initializer=initializer,
        )

        # Residual scale for memory output
        self.memory_scale = nn.Parameter(torch.tensor(0.5))

        # LM Head
        self.lm_head_proj = ComplexLinear(config.dim, config.dim, initializer=initializer)
        self.lm_head_norm = ComplexNorm(config.dim)

        self._init_weights()

    def _init_weights(self):
        embed_embeddings = {self.embed.embed_real, self.embed.embed_imag}
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module not in embed_embeddings:
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
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
    ) -> ModelOutput:
        B, L = input_ids.shape

        # 1. Embed
        z = self.embed(input_ids)
        z = self.embed_norm(z)

        # 2. Banks + Coupler per layer
        bank_z = z
        diversity_losses = []
        for i, (bank_pair, coupler, scale) in enumerate(
            zip(self.bank_pairs, self.couplers, self.bank_scales)
        ):
            residual = bank_z
            sem_out, ctx_out = bank_pair(bank_z)
            coupled = coupler(sem_out, ctx_out)
            bank_z = residual + coupled * scale

            if self.training:
                dloss = bank_pair.compute_diversity_loss(residual)
                diversity_losses.append(dloss)
                dloss_c = coupler.compute_diversity_loss()
                diversity_losses.append(dloss_c)

        # 3. SSM backbone
        ssm_out, new_state = self.ssm(bank_z, ssm_state)

        # 4. Working memory
        new_wm_keys = new_wm_values = new_wm_mask = None
        if self.working_memory is not None:
            wm_retrieved, new_wm_keys, new_wm_values, new_wm_mask = self.working_memory(
                ssm_out, wm_keys, wm_values, wm_mask,
            )

        # 5. Internal memory
        if self.internal_memory is not None:
            im_retrieved = self.internal_memory(ssm_out)

        # 6. Memory fusion (dynamic source count based on available memories)
        has_persistent = (persistent_keys is not None and persistent_mask is not None
                          and persistent_mask.sum() > 0)
        has_expert = (expert_keys is not None and expert_mask is not None
                      and expert_mask.sum() > 0)

        memory_sources = []
        if self.working_memory is not None:
            memory_sources.append(wm_retrieved)
        if self.internal_memory is not None:
            memory_sources.append(im_retrieved)

        if has_persistent:
            pm_retrieved = self.persistent_reader(
                ssm_out, persistent_keys, persistent_values, persistent_mask,
            )
            memory_sources.append(pm_retrieved)

        if has_expert:
            em_retrieved = self.expert_reader(
                ssm_out, expert_keys, expert_values, expert_mask,
            )
            memory_sources.append(em_retrieved)

        n_sources = len(memory_sources)
        if n_sources == 0:
            z_out = ssm_out
        elif n_sources == 1:
            z_out = ssm_out + memory_sources[0] * self.memory_scale
        elif n_sources == 2:
            memory_out = self.memory_fusion_2(*memory_sources)
            z_out = ssm_out + memory_out * self.memory_scale
        elif n_sources == 3:
            memory_out = self.memory_fusion_3(*memory_sources)
            z_out = ssm_out + memory_out * self.memory_scale
        else:
            memory_out = self.memory_fusion_4(*memory_sources)
            z_out = ssm_out + memory_out * self.memory_scale

        # 8. LM head
        lm = self.lm_head_proj(z_out)
        lm = self.lm_head_norm(lm)
        logits = (
            lm[..., 0] @ self.embed.embed_real.weight.T +
            lm[..., 1] @ self.embed.embed_imag.weight.T
        )

        # Diversity loss
        div_loss = None
        if diversity_losses:
            div_loss = torch.stack(diversity_losses).mean()

        return ModelOutput(
            logits=logits,
            ssm_state=new_state,
            wm_keys=new_wm_keys,
            wm_values=new_wm_values,
            wm_mask=new_wm_mask,
            diversity_loss=div_loss,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        persistent_keys: Optional[torch.Tensor] = None,
        persistent_values: Optional[torch.Tensor] = None,
        persistent_mask: Optional[torch.Tensor] = None,
        expert_keys: Optional[torch.Tensor] = None,
        expert_values: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with working memory persistence."""
        self.eval()
        generated = input_ids.clone()
        state = None
        wm_keys = wm_values = wm_mask = None

        ext = dict(
            persistent_keys=persistent_keys, persistent_values=persistent_values,
            persistent_mask=persistent_mask, expert_keys=expert_keys,
            expert_values=expert_values, expert_mask=expert_mask,
        )

        with torch.no_grad():
            out = self.forward(
                generated, ssm_state=state,
                wm_keys=wm_keys, wm_values=wm_values, wm_mask=wm_mask,
                **ext,
            )
            state = out.ssm_state
            wm_keys = out.wm_keys
            wm_values = out.wm_values
            wm_mask = out.wm_mask

            for _ in range(max_new_tokens):
                logits = out.logits[:, -1]

                if repetition_penalty != 1.0:
                    score = torch.gather(logits, 1, generated)
                    score = torch.where(
                        score > 0, score / repetition_penalty,
                        score * repetition_penalty
                    )
                    logits.scatter_(1, generated, score)

                logits = logits / temperature

                if top_k > 0:
                    v, _ = logits.topk(min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = float('-inf')

                if top_p > 0.0:
                    sorted_logits, sorted_idx = logits.sort(descending=True)
                    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    remove = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
                    sorted_logits[remove] = float('-inf')
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=1)

                out = self.forward(
                    next_token, ssm_state=state,
                    wm_keys=wm_keys, wm_values=wm_values, wm_mask=wm_mask,
                    **ext,
                )
                state = out.ssm_state
                wm_keys = out.wm_keys
                wm_values = out.wm_values
                wm_mask = out.wm_mask

        return generated

    @property
    def initializer_info(self) -> dict:
        return {
            "init_strategy": self.config.init_strategy,
            "init_seed": self.config.init_seed,
        }

    def count_parameters(self) -> Dict[str, int]:
        counts = {
            'embed (tied w/ output)': sum(p.numel() for p in self.embed.parameters()),
            'banks': sum(p.numel() for p in self.bank_pairs.parameters()),
            'couplers': sum(p.numel() for p in self.couplers.parameters()),
            'ssm': sum(p.numel() for p in self.ssm.parameters()),
            'working_memory': sum(p.numel() for p in self.working_memory.parameters()) if self.working_memory else 0,
            'internal_memory': sum(p.numel() for p in self.internal_memory.parameters()) if self.internal_memory else 0,
            'persistent_reader': sum(p.numel() for p in self.persistent_reader.parameters()),
            'expert_reader': sum(p.numel() for p in self.expert_reader.parameters()),
            'memory_fusion': (
                sum(p.numel() for p in self.memory_fusion_2.parameters()) +
                sum(p.numel() for p in self.memory_fusion_3.parameters()) +
                sum(p.numel() for p in self.memory_fusion_4.parameters())
            ),
            'lm_head_proj': (
                sum(p.numel() for p in self.lm_head_proj.parameters()) +
                sum(p.numel() for p in self.lm_head_norm.parameters())
            ),
        }
        counts['total'] = sum(counts.values())
        return counts


def create_model(config: Optional[V6Config] = None, size: str = 'small-matched') -> PhaseFieldLM:
    if config is None:
        from .config import get_config
        config = get_config(size)
    return PhaseFieldLM(config)
