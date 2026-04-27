"""
V9: PAM sequence-mixing upgrades on top of the validated V7 stack.

V7 stays frozen as the historical baseline. This module reuses V7's complex
math, CGU, trainer-compatible LM API, and presets, while swapping in a PAM
layer with optional output gating and a causal short convolution.
"""

import copy
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from v7.model import (
    AuxPredHead,
    ComplexEmbed,
    ComplexGatedUnit,
    ComplexLinear,
    ComplexNorm,
    PRESETS as V7_PRESETS,
    PhaseAssociativeLayer,
    V7Config,
    build_rope_cache,
    cabs,
    cmul,
    cnormalize,
    to_real_concat,
)


@dataclass
class V9Config(V7Config):
    # Gate the PAM readout before o_proj. Initialized as identity via
    # gate = 2 * sigmoid(0), so V9 gate runs start from V7 behavior.
    pam_output_gate: bool = False
    # Causal depthwise conv kernel before QKV projection. 0 disables it.
    pam_short_conv: int = 0


def _v9_config_from_v7(name: str, **overrides) -> V9Config:
    data = asdict(V7_PRESETS[name])
    data.update(overrides)
    return V9Config(**data)


_FLAT_CLEAN_OVERRIDES = {
    "hierarchical_dt": False,
    "dt_bias_schedule": None,
    "cross_level": False,
    "qk_norm": False,
    "multi_scale_loss": False,
    "use_reverse_assoc": False,
}


PRESETS = {
    name: _v9_config_from_v7(name)
    for name in V7_PRESETS
}
PRESETS.update({
    "medium_h16_flat": _v9_config_from_v7(
        "medium_h16_flat", **_FLAT_CLEAN_OVERRIDES,
    ),
    "medium_h16_gate": _v9_config_from_v7(
        "medium_h16_flat", **_FLAT_CLEAN_OVERRIDES, pam_output_gate=True,
    ),
    "medium_h16_conv4": _v9_config_from_v7(
        "medium_h16_flat", **_FLAT_CLEAN_OVERRIDES, pam_short_conv=4,
    ),
    "medium_h16_gate_conv4": _v9_config_from_v7(
        "medium_h16_flat", **_FLAT_CLEAN_OVERRIDES,
        pam_output_gate=True, pam_short_conv=4,
    ),
})


def get_config(preset: str = "medium_h16_flat") -> V9Config:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
    return copy.deepcopy(PRESETS[preset])


class V9PhaseAssociativeLayer(PhaseAssociativeLayer):
    """V7 PAM plus optional Mamba/RWKV-style stabilizers."""

    def __init__(self, cfg: V9Config, layer_idx: int = 0):
        super().__init__(cfg, layer_idx=layer_idx)
        self.pam_output_gate = cfg.pam_output_gate
        self.pam_short_conv = cfg.pam_short_conv

        if self.pam_output_gate:
            self.output_gate_proj = nn.Linear(cfg.dim * 2, self.inner_dim)

        if self.pam_short_conv > 0:
            self.short_conv = nn.Conv1d(
                cfg.dim * 2,
                cfg.dim * 2,
                kernel_size=self.pam_short_conv,
                padding=self.pam_short_conv - 1,
                groups=cfg.dim * 2,
            )
            self._init_short_conv_identity()

    def _init_short_conv_identity(self) -> None:
        nn.init.zeros_(self.short_conv.weight)
        nn.init.zeros_(self.short_conv.bias)
        # With left-causal slicing, the last kernel tap is the current token.
        self.short_conv.weight.data[:, 0, -1] = 1.0

    def _apply_short_conv(self, x: torch.Tensor) -> torch.Tensor:
        B, T, dim, _ = x.shape
        xf = to_real_concat(x).transpose(1, 2)
        xf = self.short_conv(xf)[..., :T].transpose(1, 2)
        return torch.stack([xf[..., :dim], xf[..., dim:]], dim=-1)

    def _output_gate(self, x: torch.Tensor, B: int, T: int, H: int, d: int) -> torch.Tensor:
        logits = self.output_gate_proj(to_real_concat(x))
        gate = 2.0 * torch.sigmoid(logits)
        return gate.view(B, T, H, d).transpose(1, 2).unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        step_offset: int = 0,
        drift_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _, _ = x.shape
        H, d = self.num_heads, self.head_dim

        x_gate = x
        if self.pam_short_conv > 0:
            x = self._apply_short_conv(x)

        # 1. Q/K/V projections
        if self.fused_qkv:
            qkv = self.qkv_proj(x).view(B, T, 3, H, d, 2)
            q = qkv[:, :, 0].transpose(1, 2).contiguous()
            k = qkv[:, :, 1].transpose(1, 2).contiguous()
            v = qkv[:, :, 2].transpose(1, 2).contiguous()
        else:
            q = self.q_proj(x).view(B, T, H, d, 2).transpose(1, 2)
            k = self.k_proj(x).view(B, T, H, d, 2).transpose(1, 2)
            v = self.v_proj(x).view(B, T, H, d, 2).transpose(1, 2)

        # 1b. Complex RoPE on Q, K
        if self.use_rope:
            end = step_offset + T
            if end > self.rope_cache.shape[0]:
                self.register_buffer(
                    "rope_cache",
                    build_rope_cache(end * 2, d).to(x.device),
                    persistent=False,
                )
            pos = self.rope_cache[step_offset:end].to(dtype=x.dtype)
            q = cmul(q, pos)
            k = cmul(k, pos)

        # 1c. Optional QK normalization
        if self.qk_norm:
            q = cnormalize(q)
            k = cnormalize(k)

        # 1d. Cross-level drift: bias Q toward higher layer's goal
        if drift_signal is not None and hasattr(self, "drift_proj"):
            drift_q = self.drift_proj(drift_signal).view(B, T, H, d, 2).transpose(1, 2)
            q = q + drift_q

        # 2. Data-dependent decay + GSP
        x_flat = to_real_concat(x)
        dt = F.softplus(self.dt_proj(x_flat) + self.dt_bias).transpose(1, 2)

        if self.use_gsp:
            p = torch.sigmoid(self.protect_gate(cabs(x))).transpose(1, 2)
            gamma = torch.exp(-dt) * (1 - p) + p
            v_prime = v * (1 - p).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = torch.exp(-dt)
            v_prime = v

        # 3a. Training: Dual Form -- no sequential loop
        _rev = self.rev_scale if self.use_reverse_assoc else None
        if state is None and T > 1:
            if self.chunk_size > 0 and T > self.chunk_size:
                y, new_state = self._forward_chunked(q, k, v_prime, gamma, d, _rev)
            else:
                scale = d ** -0.5
                q_s = q * scale
                causal = self._causal[:T, :T]
                y, new_state = self._dual_form_block(
                    q_s, k, v_prime, gamma, causal, _rev,
                )

        # 3b. Inference: Recurrent Form O(1) per token
        else:
            if state is None:
                state = torch.zeros(B, H, d, d, 2, device=x.device, dtype=x.dtype)
            scale = d ** -0.5
            y_list = []
            S = state
            for t in range(T):
                v_t = v_prime[:, :, t].unsqueeze(-2)
                k_t = k[:, :, t]
                k_conj = torch.stack([k_t[..., 0], -k_t[..., 1]], dim=-1).unsqueeze(-3)

                outer_r = v_t[..., 0] * k_conj[..., 0] - v_t[..., 1] * k_conj[..., 1]
                outer_i = v_t[..., 0] * k_conj[..., 1] + v_t[..., 1] * k_conj[..., 0]
                outer = torch.stack([outer_r, outer_i], dim=-1)

                g = gamma[:, :, t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                S = S * g + outer

                q_t = q[:, :, t].unsqueeze(-3) * scale
                sq_r = S[..., 0] * q_t[..., 0] - S[..., 1] * q_t[..., 1]
                sq_i = S[..., 0] * q_t[..., 1] + S[..., 1] * q_t[..., 0]
                y_list.append(torch.stack([sq_r, sq_i], dim=-1).sum(dim=-2))

            y = torch.stack(y_list, dim=2)
            new_state = S

        if self.pam_output_gate:
            y = y * self._output_gate(x_gate, B, T, H, d)

        # 4. Output projection + dropout
        y = y.transpose(1, 2).contiguous().view(B, T, self.inner_dim, 2)
        out = self.o_proj(y)

        if self.training:
            mask = self.dropout(torch.ones(B, T, self.dim, device=x.device))
            out = out * mask.unsqueeze(-1)

        return out, new_state


class V9Block(nn.Module):
    """Pre-norm residual: V7 CGU plus upgraded PAM sequence mixer."""

    def __init__(self, cfg: V9Config, layer_idx: int = 0):
        super().__init__()
        self.norm1 = ComplexNorm(cfg.dim)
        self.cgu = ComplexGatedUnit(cfg.dim, cfg.expand, activation=cfg.activation)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.cgu_dropout = nn.Dropout(cfg.dropout)
        self.norm2 = ComplexNorm(cfg.dim)
        self.pam = V9PhaseAssociativeLayer(cfg, layer_idx=layer_idx)
        self.pam_scale = nn.Parameter(torch.tensor(0.1))
        self._dim = cfg.dim

    def forward(
        self,
        x: torch.Tensor,
        pam_state: Optional[torch.Tensor] = None,
        step_offset: int = 0,
        drift_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cgu_out = self.cgu(self.norm1(x))
        if self.training:
            drop_mask = self.cgu_dropout(
                torch.ones(cgu_out.shape[:-1], device=cgu_out.device)
            )
            cgu_out = cgu_out * drop_mask.unsqueeze(-1)
        x = x + cgu_out * self.cgu_scale
        pam_out, new_state = self.pam(
            self.norm2(x), state=pam_state, step_offset=step_offset,
            drift_signal=drift_signal,
        )
        x = x + pam_out * self.pam_scale
        return x, new_state, pam_out


class V9LM(nn.Module):
    """
    ComplexEmbed -> [V9Block] x N -> Tied Complex LM Head.

    The public API matches V7LM so the existing trainer/checkpoint path can be
    reused for apples-to-apples experiments.
    """

    def __init__(self, cfg: V9Config):
        super().__init__()
        if cfg.hierarchical_dt and cfg.dt_bias_schedule is None:
            cfg.dt_bias_schedule = tuple(
                torch.linspace(-6.91, 0.0, cfg.n_layers).tolist()
            )
        self.config = cfg

        self.embed = ComplexEmbed(cfg.vocab_size, cfg.dim)
        self.embed_norm = ComplexNorm(cfg.dim)
        self.blocks = nn.ModuleList([
            V9Block(cfg, layer_idx=i) for i in range(cfg.n_layers)
        ])
        self.output_norm = ComplexNorm(cfg.dim)
        self.lm_head_proj = ComplexLinear(cfg.dim, cfg.dim)
        self.lm_head_norm = ComplexNorm(cfg.dim)

        self.aux_heads = nn.ModuleDict()
        self.aux_offsets: Dict[int, int] = {}
        if cfg.multi_scale_loss:
            aux_indices = list(range(0, cfg.n_layers, cfg.aux_layer_stride))
            for idx in aux_indices:
                frac = 1.0 - idx / max(cfg.n_layers - 1, 1)
                offset = max(1, int(cfg.max_aux_offset * frac))
                self.aux_heads[str(idx)] = AuxPredHead(cfg.dim)
                self.aux_offsets[idx] = offset

        self._init_weights()

    def _init_weights(self):
        """Match V7 init, then restore identity-initialized V9 additions."""
        embed_embeddings = {self.embed.embed_real, self.embed.embed_imag}
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module not in embed_embeddings:
                nn.init.normal_(module.weight, std=0.02)
        self._reinit_custom_biases()

    def _reinit_custom_biases(self):
        for module in self.modules():
            if hasattr(module, "protect_gate") and isinstance(module.protect_gate, nn.Linear):
                nn.init.constant_(module.protect_gate.bias, -3.0)
            if hasattr(module, "output_gate_proj"):
                nn.init.zeros_(module.output_gate_proj.weight)
                nn.init.zeros_(module.output_gate_proj.bias)
            if hasattr(module, "short_conv"):
                module._init_short_conv_identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        step_offset: int = 0,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        z = self.embed_norm(self.embed(input_ids))

        use_ckpt = (
            self.config.gradient_checkpointing
            and self.training
            and states is None
        )

        new_states = []
        drift = None
        aux_loss = torch.tensor(0.0, device=input_ids.device)
        n_layers = self.config.n_layers

        for i, block in enumerate(self.blocks):
            s = states[i] if states is not None else None
            if use_ckpt:
                z, new_s, pam_out = self._checkpointed_block(
                    block, z, step_offset, drift,
                )
            else:
                z, new_s, pam_out = block(
                    z, pam_state=s, step_offset=step_offset, drift_signal=drift,
                )
            new_states.append(new_s)
            drift = pam_out if self.config.cross_level else None

            if labels is not None and str(i) in self.aux_heads:
                offset = self.aux_offsets[i]
                if offset < labels.shape[1]:
                    aux_logits = self.aux_heads[str(i)](
                        z, self.embed.embed_real.weight, self.embed.embed_imag.weight,
                    )
                    shifted = labels[:, offset:]
                    trimmed = aux_logits[:, :shifted.shape[1]]
                    if trimmed.numel() > 0:
                        layer_loss = F.cross_entropy(
                            trimmed.reshape(-1, trimmed.size(-1)),
                            shifted.reshape(-1),
                        )
                        frac = i / max(n_layers - 1, 1)
                        w = math.exp(-2.0 * (1.0 - frac))
                        aux_loss = aux_loss + w * layer_loss

        z = self.output_norm(z)
        lm = self.lm_head_norm(self.lm_head_proj(z))
        logits = (
            lm[..., 0] @ self.embed.embed_real.weight.T
            + lm[..., 1] @ self.embed.embed_imag.weight.T
        )
        return logits, new_states, aux_loss

    @staticmethod
    def _checkpointed_block(
        block: V9Block,
        z: torch.Tensor,
        step_offset: int,
        drift: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def run_block(z_in, *drift_args):
            d = drift_args[0] if drift_args else None
            return block(z_in, pam_state=None, step_offset=step_offset, drift_signal=d)

        if drift is not None:
            return grad_checkpoint(run_block, z, drift, use_reentrant=False)
        return grad_checkpoint(run_block, z, use_reentrant=False)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()

        logits, states, _ = self.forward(generated)
        step = generated.shape[1]

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1] / temperature

            if repetition_penalty != 1.0:
                score = torch.gather(next_logits, 1, generated)
                score = torch.where(
                    score > 0, score / repetition_penalty,
                    score * repetition_penalty,
                )
                next_logits.scatter_(1, generated, score)

            if top_k > 0:
                v, _ = next_logits.topk(min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, -1:]] = float("-inf")

            if top_p > 0:
                sorted_logits, sorted_idx = next_logits.sort(descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove = cum_probs - sorted_logits.softmax(dim=-1) >= top_p
                sorted_logits[remove] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            next_token = torch.multinomial(next_logits.softmax(dim=-1), 1)
            generated = torch.cat([generated, next_token], dim=1)

            logits, states, _ = self.forward(
                next_token, states=states, step_offset=step,
            )
            step += 1

        return generated

    def count_parameters(self) -> Dict[str, int]:
        embed_p = sum(p.numel() for p in self.embed.parameters())
        block_p = sum(p.numel() for b in self.blocks for p in b.parameters())
        head_p = (
            sum(p.numel() for p in self.lm_head_proj.parameters())
            + sum(p.numel() for p in self.lm_head_norm.parameters())
        )
        norm_p = (
            sum(p.numel() for p in self.embed_norm.parameters())
            + sum(p.numel() for p in self.output_norm.parameters())
        )
        aux_p = sum(p.numel() for p in self.aux_heads.parameters())
        total = embed_p + block_p + head_p + norm_p + aux_p
        result: Dict[str, int] = {
            "embedding (tied)": embed_p,
            "blocks": block_p,
            "norms": norm_p,
            "lm_head": head_p,
            "total": total,
        }
        if aux_p > 0:
            result["aux_heads"] = aux_p
        return result
