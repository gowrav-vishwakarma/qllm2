r"""V8 Language Model.

Architecture
============

::

    Tokens
      |
      v
    ComplexEmbed (V7)
      |
      v
    [V7Block] x N        <-- the QPAM "grammar" backbone (reused unchanged)
      |
      v
    Output ComplexNorm
      |
      v
    QuantumLogicCore     <-- the new piece: Probe -> Sasaki -> OrthoHalt loop
      |
      v
    Tied complex LM head -> logits

When ``QLCConfig.enabled = False`` the QLC step is skipped and the model is
mathematically identical to a V7LM with the same backbone -- this is the
"V8-A passthrough" row in the ablation matrix and the sanity gate after
Stage A pretraining.

Backbone freezing (Stage B) is implemented by ``freeze_backbone()`` which
toggles ``requires_grad`` and switches BatchNorm-style modules to eval mode.

Logits return signature mirrors :class:`v7.model.V7LM` so existing trainer
loops work unchanged: ``(logits, states, aux_loss)``. ``aux_loss`` carries
``ponder_lambda * ponder_cost`` from the QLC plus any backbone aux losses.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from v8.backbone import (
    V7Config, V7Block, ComplexEmbed, ComplexNorm, ComplexLinear,
)
from v8.config import V8Config, QLCConfig
from v8.qlc.reason_loop import QuantumLogicCore, QLCDiagnostics


class V8LM(nn.Module):
    """V8: V7 QPAM backbone + Quantum-Logic Core."""

    def __init__(self, cfg: V8Config):
        super().__init__()
        self.config = cfg
        bcfg = cfg.backbone
        self.backbone_cfg = bcfg
        self.qlc_cfg = cfg.qlc

        # Backbone (verbatim V7Block stack — no PAM logic re-implemented here).
        self.embed = ComplexEmbed(bcfg.vocab_size, bcfg.dim)
        self.embed_norm = ComplexNorm(bcfg.dim)
        self.blocks = nn.ModuleList([
            V7Block(bcfg, layer_idx=i) for i in range(bcfg.n_layers)
        ])
        self.output_norm = ComplexNorm(bcfg.dim)

        # Quantum-Logic Core (or identity passthrough).
        if cfg.qlc.enabled:
            self.qlc: Optional[QuantumLogicCore] = QuantumLogicCore(
                dim=bcfg.dim,
                rank=cfg.qlc.rank,
                bank_size=cfg.qlc.bank_size,
                top_k=cfg.qlc.top_k,
                t_max=cfg.qlc.t_max,
                n_heads=cfg.qlc.n_heads,
                ponder_lambda=cfg.qlc.ponder_lambda,
                bank_temperature=cfg.qlc.bank_temperature,
                quantale_off=cfg.qlc.quantale_off,
                orthohalt_off=cfg.qlc.orthohalt_off,
                qr_refresh_every=cfg.qlc.qr_refresh_every,
                use_complex=cfg.qlc.use_complex,
                out_scale_init=cfg.qlc.out_scale_init,
                out_scale_learnable=cfg.qlc.out_scale_learnable,
                renormalize_psi=cfg.qlc.renormalize_psi,
                halt_mode=cfg.qlc.halt_mode,
                unsharp_target=cfg.qlc.unsharp_target,
                quantale_order_test=cfg.qlc.quantale_order_test,
            )
        else:
            self.qlc = None

        # LM head (mirrors V7).
        self.lm_head_proj = ComplexLinear(bcfg.dim, bcfg.dim)
        self.lm_head_norm = ComplexNorm(bcfg.dim)

        self._init_weights()

        if cfg.freeze_backbone:
            self.freeze_backbone(unfreeze_lm_head=cfg.unfreeze_lm_head)

        # Cache of last QLC diagnostics for the trainer to log.
        self._last_qlc_diag: Optional[QLCDiagnostics] = None

    # ── Init / freezing ────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        embed_embeddings = {self.embed.embed_real, self.embed.embed_imag}
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and module not in embed_embeddings:
                nn.init.normal_(module.weight, std=0.02)
        # Re-apply custom biases (protect_gate, ortho-halt cls_head identity).
        for _, module in self.named_modules():
            if hasattr(module, 'protect_gate') and isinstance(module.protect_gate, nn.Linear):
                nn.init.constant_(module.protect_gate.bias, -3.0)
        if self.qlc is not None:
            with torch.no_grad():
                head = getattr(self.qlc.halt, 'cls_head', None)
                # Only re-init when the head is the canonical OrthoHalt 3x3
                # mapping. Empirical halts (DeltaHalt, EntropyHalt) ship with
                # their own carefully-shaped cls_head and must be left alone.
                if (
                    head is not None
                    and isinstance(head, nn.Linear)
                    and head.weight.shape == (3, 3)
                ):
                    head.weight.copy_(torch.eye(3) * 4.0)
                    if head.bias is not None:
                        head.bias.zero_()

    def freeze_backbone(self, unfreeze_lm_head: bool = False) -> None:
        """Freeze the embed / V7Block stack / output_norm (and LM head) for Stage B.

        QLC parameters remain trainable. By default (matching the original
        Stage B behaviour) the LM head also stays frozen so the
        backbone-vs-QLC contribution is unambiguously attributable.

        Set ``unfreeze_lm_head=True`` (used by the
        ``stageB_lmhead_unfrozen`` preset, rethink-plan §6) to keep
        ``lm_head_proj`` + ``lm_head_norm`` trainable. The token embedding is
        still frozen (it is shared with the LM head via tied weights and we
        want the bottleneck to be the *projection*, not the vocab matrix).
        """
        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.embed_norm.parameters():
            p.requires_grad = False
        for p in self.blocks.parameters():
            p.requires_grad = False
        for p in self.output_norm.parameters():
            p.requires_grad = False
        if not unfreeze_lm_head:
            for p in self.lm_head_proj.parameters():
                p.requires_grad = False
            for p in self.lm_head_norm.parameters():
                p.requires_grad = False
        # Belt-and-braces: also disable gradient checkpointing on the backbone
        # (it's wasted work when the params are frozen and produces noisy autograd
        # warnings on AMP).
        self.backbone_cfg.gradient_checkpointing = False

    def unfreeze_backbone(self) -> None:
        """Restore trainability for Stage C joint fine-tuning."""
        for p in self.embed.parameters():
            p.requires_grad = True
        for p in self.embed_norm.parameters():
            p.requires_grad = True
        for p in self.blocks.parameters():
            p.requires_grad = True
        for p in self.output_norm.parameters():
            p.requires_grad = True
        for p in self.lm_head_proj.parameters():
            p.requires_grad = True
        for p in self.lm_head_norm.parameters():
            p.requires_grad = True

    # ── Backbone-only forward (for KL anchor in Stage C) ───────────────────

    def encode_backbone_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the embedding + V7Block stack + output_norm and return the
        complex hidden state ``[B, T, d, 2]`` that feeds the QLC / LM head.

        This is the canonical "psi" consumed by the EffectAlgebraBank. Used
        by the InfoNCE auxiliary in Stage B (rethink-plan §5) to score
        cloze-style queries against the bank's effects.

        Unlike :meth:`backbone_logits`, this method *does* allow gradients
        to flow back through the backbone -- the caller is responsible for
        zeroing them if the backbone is frozen (Stage B). The bank's own
        parameters always receive gradients regardless.
        """
        z = self.embed_norm(self.embed(input_ids))
        drift = None
        for block in self.blocks:
            z, _, pam_out = block(z, pam_state=None, step_offset=0, drift_signal=drift)
            drift = pam_out if self.backbone_cfg.cross_level else None
        z = self.output_norm(z)
        return z

    @torch.no_grad()
    def backbone_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits with the QLC bypassed (Stage A behaviour).

        Used to provide the KL-anchor target during Stage C joint fine-tuning.
        Always runs in eval mode and does not track gradients.
        """
        was_training = self.training
        self.eval()
        try:
            z = self.embed_norm(self.embed(input_ids))
            for block in self.blocks:
                z, _, _ = block(z, pam_state=None, step_offset=0, drift_signal=None)
            z = self.output_norm(z)
            lm = self.lm_head_norm(self.lm_head_proj(z))
            logits = (
                lm[..., 0] @ self.embed.embed_real.weight.T
                + lm[..., 1] @ self.embed.embed_imag.weight.T
            )
            return logits
        finally:
            if was_training:
                self.train()

    # ── Main forward ───────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        step_offset: int = 0,
        labels: Optional[torch.Tensor] = None,
        return_qlc_diag: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Forward pass.

        Returns
        -------
        logits : ``[B, T, vocab]``
        new_states : list of per-layer PAM states (mirrors V7LM)
        aux_loss : scalar -- ponder_lambda * ponder_cost (zero when QLC off)
        """
        z = self.embed_norm(self.embed(input_ids))

        new_states: List[torch.Tensor] = []
        drift = None
        for i, block in enumerate(self.blocks):
            s = states[i] if states is not None else None
            z, new_s, pam_out = block(
                z, pam_state=s, step_offset=step_offset, drift_signal=drift,
            )
            new_states.append(new_s)
            drift = pam_out if self.backbone_cfg.cross_level else None

        z = self.output_norm(z)

        # QLC reasoning loop (or passthrough).
        ponder_cost = torch.tensor(0.0, device=z.device, dtype=z.dtype if z.is_floating_point() else torch.float32)
        if self.qlc is not None:
            z, ponder_cost, diag = self.qlc(z, return_diagnostics=return_qlc_diag)
            if return_qlc_diag:
                self._last_qlc_diag = diag

        lm = self.lm_head_norm(self.lm_head_proj(z))
        logits = (
            lm[..., 0] @ self.embed.embed_real.weight.T
            + lm[..., 1] @ self.embed.embed_imag.weight.T
        )

        aux_loss = self.qlc_cfg.ponder_lambda * ponder_cost if self.qlc is not None else ponder_cost
        return logits, new_states, aux_loss

    # ── Generation (mirrors V7LM.generate) ─────────────────────────────────

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
            next_logits = logits[:, -1] / max(temperature, 1e-6)

            if repetition_penalty != 1.0:
                score = torch.gather(next_logits, 1, generated)
                score = torch.where(
                    score > 0, score / repetition_penalty,
                    score * repetition_penalty,
                )
                next_logits.scatter_(1, generated, score)

            if top_k > 0:
                v, _ = next_logits.topk(min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, -1:]] = float('-inf')

            if top_p > 0:
                sorted_logits, sorted_idx = next_logits.sort(descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove = cum_probs - sorted_logits.softmax(dim=-1) >= top_p
                sorted_logits[remove] = float('-inf')
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            next_token = torch.multinomial(next_logits.softmax(dim=-1), 1)
            generated = torch.cat([generated, next_token], dim=1)

            logits, states, _ = self.forward(
                next_token, states=states, step_offset=step,
            )
            step += 1

        return generated

    # ── Diagnostics ────────────────────────────────────────────────────────

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
        qlc_p = sum(p.numel() for p in self.qlc.parameters()) if self.qlc is not None else 0
        total = embed_p + block_p + head_p + norm_p + qlc_p
        out = {
            'embedding (tied)': embed_p,
            'backbone_blocks': block_p,
            'norms': norm_p,
            'lm_head': head_p,
            'qlc': qlc_p,
            'total': total,
        }
        return out

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def last_qlc_diagnostics(self) -> Optional[QLCDiagnostics]:
        return self._last_qlc_diag


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def load_backbone_from_v7_checkpoint(
    model: V8LM,
    ckpt_path: str,
    strict: bool = False,
    map_location: str = "cpu",
) -> Dict[str, list]:
    """Load a V7 checkpoint's backbone weights into a V8LM in place.

    The Stage A trainer saves V8 checkpoints in V7-compatible layout (the
    backbone keys are the V7Block stack + embed + lm_head) so this function
    is also the loader for ``v8/checkpoints/qpam_stageA.pt`` when starting
    Stage B. QLC parameters are *not* touched.

    Returns the (missing_keys, unexpected_keys) report from
    :meth:`load_state_dict` so the caller can verify exactly which tensors
    were filled in.
    """
    state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip any "_orig_mod." prefix from torch.compile.
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # Filter to backbone keys only.
    backbone_prefixes = (
        "embed.", "embed_norm.", "blocks.", "output_norm.",
        "lm_head_proj.", "lm_head_norm.",
    )
    filtered = {k: v for k, v in state.items() if k.startswith(backbone_prefixes)}
    info = model.load_state_dict(filtered, strict=False)
    return {"missing": list(info.missing_keys), "unexpected": list(info.unexpected_keys)}
