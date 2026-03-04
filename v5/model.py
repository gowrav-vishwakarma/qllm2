"""
V5 Algebraic Language Model.

Architecture:
    Tokens -> ComplexEmbed -> [AlgebraicBank + ComplexSSM + PhaseAttn] x N -> LM Head

Every operation preserves complex algebraic structure until the final
projection to vocabulary logits.

No separate FFN -- following CliffordNet's finding that algebraic
interactions make FFN layers redundant.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .config import V5Config
from .core.complex import (
    ComplexEmbed, ComplexLinear, ComplexNorm, to_real,
)
from .core.ssm import ComplexSSM, SSMState
from .core.attention import PhaseAttention
from .core.bank import MultiBank


@dataclass
class ModelOutput:
    logits: torch.Tensor                    # [B, L, vocab_size]
    ssm_state: Optional[SSMState] = None
    diversity_loss: Optional[torch.Tensor] = None


class AlgebraicLM(nn.Module):
    """
    V5 Algebraic Language Model.

    Each layer block:
        z -> Norm -> MultiBank -> (+residual) -> z'
        z' -> SSM layer i -> (+residual) -> z''
        [Every K layers: z'' -> Norm -> PhaseAttention -> (+residual)]

    The SSM is a full stacked backbone (handles its own norms/residuals).
    Banks and attention are interleaved around it.
    """

    def __init__(self, config: V5Config):
        super().__init__()
        self.config = config

        # Embedding
        self.embed = ComplexEmbed(config.vocab_size, config.dim)
        self.embed_norm = ComplexNorm(config.dim)

        # Per-layer banks (before SSM)
        self.banks = nn.ModuleList([
            MultiBank(config.dim, config.num_banks, config.bank_expand, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.bank_norms = nn.ModuleList([
            ComplexNorm(config.dim)
            for _ in range(config.num_layers)
        ])
        self.bank_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0))
            for _ in range(config.num_layers)
        ])

        # SSM backbone (stacked, handles its own residuals)
        self.ssm = ComplexSSM(
            dim=config.dim,
            state_dim=config.state_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        # Sparse attention layers (every K-th layer)
        self.attn_layers = nn.ModuleDict()
        self.attn_norms = nn.ModuleDict()
        self.attn_scales = nn.ParameterDict()
        if config.attn_every_k > 0:
            for i in range(config.attn_every_k - 1, config.num_layers, config.attn_every_k):
                key = str(i)
                self.attn_layers[key] = PhaseAttention(
                    config.dim, config.num_heads, config.window_size, config.dropout,
                )
                self.attn_norms[key] = ComplexNorm(config.dim)
                self.attn_scales[key] = nn.Parameter(torch.tensor(0.1))

        # LM Head: complex -> real -> vocab logits
        self.lm_head_proj = ComplexLinear(config.dim, config.dim)
        self.lm_head_norm = ComplexNorm(config.dim)
        self.output_proj = nn.Linear(config.dim * 2, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,          # [B, L]
        ssm_state: Optional[SSMState] = None,
    ) -> ModelOutput:
        B, L = input_ids.shape

        # 1. Embed to complex space
        z = self.embed(input_ids)          # [B, L, dim, 2]
        z = self.embed_norm(z)

        # 2. Pre-SSM: apply banks per layer
        #    We run banks sequentially to match SSM layer count,
        #    accumulating the bank-processed signal before feeding SSM.
        bank_z = z
        diversity_losses = []
        for i, (bank, norm, scale) in enumerate(
            zip(self.banks, self.bank_norms, self.bank_scales)
        ):
            residual = bank_z
            bank_out = bank(norm(bank_z))
            bank_z = residual + bank_out * scale

            if self.training and self.config.num_banks >= 2:
                dloss = bank.compute_diversity_loss(norm(residual))
                if dloss is not None:
                    diversity_losses.append(dloss)

        # 3. SSM backbone (stacked, parallel scan)
        ssm_out, new_state = self.ssm(bank_z, ssm_state)

        # 4. Post-SSM: sparse attention
        z_out = ssm_out
        for key in self.attn_layers:
            norm = self.attn_norms[key]
            attn = self.attn_layers[key]
            scale = self.attn_scales[key]
            residual = z_out
            z_out = residual + attn(norm(z_out)) * scale

        # 5. LM head: complex -> real -> logits
        lm = self.lm_head_proj(z_out)
        lm = self.lm_head_norm(lm)
        lm_real = to_real(lm, mode='concat')  # [B, L, dim*2]
        logits = self.output_proj(lm_real)      # [B, L, vocab_size]

        # Aggregate diversity loss
        div_loss = None
        if diversity_losses:
            div_loss = torch.stack(diversity_losses).mean()

        return ModelOutput(
            logits=logits,
            ssm_state=new_state,
            diversity_loss=div_loss,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Simple greedy/sampling generation."""
        self.eval()
        generated = input_ids.clone()
        state = None

        with torch.no_grad():
            # Process prompt
            out = self.forward(generated, ssm_state=state)
            state = out.ssm_state

            for _ in range(max_new_tokens):
                logits = out.logits[:, -1] / temperature

                if top_k > 0:
                    v, _ = logits.topk(top_k)
                    logits[logits < v[:, -1:]] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=1)

                # Next step: only process the new token
                out = self.forward(next_token, ssm_state=state)
                state = out.ssm_state

        return generated

    def count_parameters(self) -> Dict[str, int]:
        """Parameter count by component."""
        counts = {
            'embed': sum(p.numel() for p in self.embed.parameters()),
            'banks': sum(p.numel() for p in self.banks.parameters()),
            'ssm': sum(p.numel() for p in self.ssm.parameters()),
            'attention': sum(
                p.numel()
                for key in self.attn_layers
                for p in self.attn_layers[key].parameters()
            ),
            'lm_head': (
                sum(p.numel() for p in self.lm_head_proj.parameters()) +
                sum(p.numel() for p in self.lm_head_norm.parameters()) +
                sum(p.numel() for p in self.output_proj.parameters())
            ),
        }
        counts['total'] = sum(counts.values())
        return counts


def create_model(config: Optional[V5Config] = None, size: str = 'small') -> AlgebraicLM:
    if config is None:
        from .config import get_config
        config = get_config(size)
    return AlgebraicLM(config)
