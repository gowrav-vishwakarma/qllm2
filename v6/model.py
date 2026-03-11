"""
V6 Phase-First Language Model.

Architecture (attention-free by default, optional sparse PhaseAttention):
    Tokens -> ComplexEmbed
    -> [NamedBankPair -> PhaseInterferenceCoupler (-> PhaseAttention?)] x N layers
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
from .core.ssm import SSMState
from .backbone import PhaseFieldBackbone, BackboneOutput, MemoryFusion


@dataclass
class ModelOutput:
    logits: torch.Tensor
    ssm_state: Optional[SSMState] = None
    wm_keys: Optional[torch.Tensor] = None
    wm_values: Optional[torch.Tensor] = None
    wm_mask: Optional[torch.Tensor] = None
    diversity_loss: Optional[torch.Tensor] = None


class PhaseFieldLM(nn.Module):
    """
    V6 Phase-First Language Model.

    Attention-free by default. Long-context coherence via:
    1. Multi-timescale SSM (fast/medium/slow decay lanes)
    2. Working memory (learned write/read, non-decaying per-sequence)
    3. Internal memory (trained slots, general knowledge)
    4. Optional persistent memory (per-user, loaded from disk)
    5. Optional sparse PhaseAttention (disabled by default)
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

        # Shared backbone (banks + couplers + SSM + memory)
        self.backbone = PhaseFieldBackbone(config, initializer)

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

        # 2. Backbone (banks + SSM + memory)
        bb = self.backbone(
            z, ssm_state=ssm_state,
            wm_keys=wm_keys, wm_values=wm_values, wm_mask=wm_mask,
            persistent_keys=persistent_keys, persistent_values=persistent_values,
            persistent_mask=persistent_mask,
            expert_keys=expert_keys, expert_values=expert_values,
            expert_mask=expert_mask,
        )

        # 3. LM head
        lm = self.lm_head_proj(bb.z_out)
        lm = self.lm_head_norm(lm)
        logits = (
            lm[..., 0] @ self.embed.embed_real.weight.T +
            lm[..., 1] @ self.embed.embed_imag.weight.T
        )

        return ModelOutput(
            logits=logits,
            ssm_state=bb.ssm_state,
            wm_keys=bb.wm_keys,
            wm_values=bb.wm_values,
            wm_mask=bb.wm_mask,
            diversity_loss=bb.diversity_loss,
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
        bb_counts = self.backbone.count_parameters()
        counts = {
            'embed (tied w/ output)': sum(p.numel() for p in self.embed.parameters()),
            **bb_counts,
            'lm_head_proj': (
                sum(p.numel() for p in self.lm_head_proj.parameters()) +
                sum(p.numel() for p in self.lm_head_norm.parameters())
            ),
        }
        counts['total'] = sum(counts.values())
        return counts


def create_model(config: Optional[V6Config] = None, size: str = 'small-matched'):
    """Factory: returns the right model class based on config.mode."""
    if config is None:
        from .config import get_config
        config = get_config(size)
    mode = getattr(config, 'mode', 'autoregressive')
    if mode == 'autoregressive':
        return PhaseFieldLM(config)
    elif mode == 'two_pass':
        from .two_pass_model import TwoPassLM
        return TwoPassLM(config)
    else:
        from .diffusion_model import PhaseFieldDiffusion
        return PhaseFieldDiffusion(config)
