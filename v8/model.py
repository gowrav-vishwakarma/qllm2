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
from typing import Iterator, List, Optional, Tuple, Dict

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
        self.qlc_insert_layers = tuple(sorted(set(cfg.qlc_insert_layers)))
        self.shared_interleaved_qlc = bool(cfg.shared_interleaved_qlc)
        self.final_qlc_enabled = bool(cfg.final_qlc_enabled)

        # Backbone (verbatim V7Block stack — no PAM logic re-implemented here).
        self.embed = ComplexEmbed(bcfg.vocab_size, bcfg.dim)
        self.embed_norm = ComplexNorm(bcfg.dim)
        self.blocks = nn.ModuleList([
            V7Block(bcfg, layer_idx=i) for i in range(bcfg.n_layers)
        ])
        self.output_norm = ComplexNorm(bcfg.dim)

        # Quantum-Logic Core(s). Empty qlc_insert_layers preserves legacy V8:
        # one post-backbone QLC. V8.3 can inject a shared QLC between selected
        # blocks and optionally disable the final post-backbone adapter.
        self.interleaved_qlc: Optional[QuantumLogicCore] = None
        self.interleaved_qlcs = nn.ModuleDict()
        if cfg.qlc.enabled and self.qlc_insert_layers:
            if self.shared_interleaved_qlc:
                self.interleaved_qlc = self._build_qlc(
                    out_scale_init=cfg.interleaved_out_scale,
                )
            else:
                for layer_idx in self.qlc_insert_layers:
                    self.interleaved_qlcs[str(layer_idx)] = self._build_qlc(
                        out_scale_init=cfg.interleaved_out_scale,
                    )
            self.qlc: Optional[QuantumLogicCore] = (
                self._build_qlc() if self.final_qlc_enabled else None
            )
        elif cfg.qlc.enabled:
            self.qlc = self._build_qlc()
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
        self._last_qlc_diags: List[QLCDiagnostics] = []

    def _build_qlc(self, *, out_scale_init: Optional[float] = None) -> QuantumLogicCore:
        q = self.qlc_cfg
        return QuantumLogicCore(
            dim=self.backbone_cfg.dim,
            rank=q.rank,
            bank_size=q.bank_size,
            top_k=q.top_k,
            t_max=q.t_max,
            n_heads=q.n_heads,
            ponder_lambda=q.ponder_lambda,
            bank_temperature=q.bank_temperature,
            quantale_off=q.quantale_off,
            orthohalt_off=q.orthohalt_off,
            qr_refresh_every=q.qr_refresh_every,
            use_complex=q.use_complex,
            out_scale_init=q.out_scale_init if out_scale_init is None else out_scale_init,
            out_scale_learnable=q.out_scale_learnable,
            renormalize_psi=q.renormalize_psi,
            halt_mode=q.halt_mode,
            unsharp_target=q.unsharp_target,
            quantale_order_test=q.quantale_order_test,
            use_iteration_context=q.use_iteration_context,
            use_layer_context=q.use_layer_context,
            rope_conditioned_probe=q.rope_conditioned_probe,
            weighted_projector=q.weighted_projector,
            max_reason_iters=q.max_reason_iters,
            max_context_layers=q.max_context_layers,
        )

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
        for qlc in self.iter_qlc_modules():
            with torch.no_grad():
                head = getattr(qlc.halt, 'cls_head', None)
                # Only re-init when the head is the canonical OrthoHalt 3x3
                # mapping. Empirical halts (DeltaHalt, EntropyHalt) ship with
                # their own carefully-shaped cls_head and must be left alone.
                #
                # Continue-biased init (v8.2): the legacy 4*I init forced the
                # softmax to halt-no on iter 1 because beta dominates at random
                # init. With weight=eye*1.0 and bias=[0,0,2] the default
                # distribution is roughly [0.10, 0.10, 0.80] before any signal
                # is learned, so the loop runs all t_max iterations from step
                # 0 and gradients flow back through every iteration.
                if (
                    head is not None
                    and isinstance(head, nn.Linear)
                    and head.weight.shape == (3, 3)
                ):
                    head.weight.copy_(torch.eye(3) * 1.0)
                    if head.bias is not None:
                        head.bias.copy_(torch.tensor(
                            [0.0, 0.0, 2.0], dtype=head.bias.dtype,
                            device=head.bias.device,
                        ))
                # Also reset the unsharp target gate (v8.2): default sigmoid(1.5)
                # ~= 0.82 routes most amp_sq to alpha. The legacy zero init
                # split it 50/50 with gamma which kept both at noise floor.
                gate = getattr(qlc.halt, 'target_gate', None)
                if isinstance(gate, nn.Parameter) and gate.dim() == 1:
                    gate.fill_(1.5)

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

    def iter_qlc_modules(self) -> Iterator[QuantumLogicCore]:
        """Yield active QLC modules once each, including interleaved modules."""
        seen = set()
        modules: List[Optional[QuantumLogicCore]] = [self.qlc, self.interleaved_qlc]
        modules.extend(self.interleaved_qlcs.values())
        for module in modules:
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            yield module

    def primary_qlc(self) -> Optional[QuantumLogicCore]:
        """Return the bank-owning QLC used by diagnostics / InfoNCE."""
        if self.interleaved_qlc is not None:
            return self.interleaved_qlc
        if len(self.interleaved_qlcs) > 0:
            first_key = sorted(self.interleaved_qlcs.keys(), key=int)[0]
            return self.interleaved_qlcs[first_key]
        return self.qlc

    def has_qlc(self) -> bool:
        return self.primary_qlc() is not None

    def _qlc_for_insert_layer(self, layer_idx: int) -> Optional[QuantumLogicCore]:
        if layer_idx not in self.qlc_insert_layers:
            return None
        if self.interleaved_qlc is not None:
            return self.interleaved_qlc
        return self.interleaved_qlcs.get(str(layer_idx))

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
        ponder_cost = torch.tensor(
            0.0,
            device=z.device,
            dtype=z.dtype if z.is_floating_point() else torch.float32,
        )
        align_signals: List[torch.Tensor] = []
        qlc_diags: List[QLCDiagnostics] = []
        prev_qlc_residual: Optional[torch.Tensor] = None
        for i, block in enumerate(self.blocks):
            s = states[i] if states is not None else None
            z, new_s, pam_out = block(
                z, pam_state=s, step_offset=step_offset, drift_signal=drift,
            )
            new_states.append(new_s)
            drift = pam_out if self.backbone_cfg.cross_level else None

            qlc_at_layer = self._qlc_for_insert_layer(i)
            if qlc_at_layer is not None:
                z_before_qlc = z
                z, pc, align_signal, diag = qlc_at_layer(
                    z,
                    return_diagnostics=return_qlc_diag,
                    return_align=True,
                    layer_idx=i,
                    position_offset=step_offset,
                    insertion_point=f"layer_{i}",
                )
                ponder_cost = ponder_cost + pc
                align_signals.append(align_signal)
                if return_qlc_diag and diag is not None:
                    residual = z - z_before_qlc
                    with torch.no_grad():
                        delta = (
                            residual[..., 0].square() + residual[..., 1].square()
                        ).sum(dim=-1).mean()
                        diag.downstream_delta_l2 = float(delta.item())
                        if prev_qlc_residual is not None:
                            r0 = prev_qlc_residual.reshape(-1, 2)
                            r1 = residual.reshape(-1, 2)
                            dot = (r0 * r1).sum()
                            n0 = r0.square().sum().sqrt()
                            n1 = r1.square().sum().sqrt()
                            diag.memory_echo_cos = float(
                                (dot / (n0 * n1).clamp_min(1e-12)).item()
                            )
                        prev_qlc_residual = residual.detach()
                    qlc_diags.append(diag)

        z = self.output_norm(z)

        # Final post-backbone QLC reasoning loop (legacy placement) or passthrough.
        if self.qlc is not None:
            # Always request align_signal so the v8.2 alignment auxiliary can
            # flow into aux_loss. Returns a 4-tuple in this branch.
            z_before_qlc = z
            z, pc, align_signal, diag = self.qlc(
                z,
                return_diagnostics=return_qlc_diag,
                return_align=True,
                layer_idx=None,
                position_offset=step_offset,
                insertion_point="final",
            )
            ponder_cost = ponder_cost + pc
            align_signals.append(align_signal)
            if return_qlc_diag:
                if diag is not None:
                    residual = z - z_before_qlc
                    with torch.no_grad():
                        delta = (
                            residual[..., 0].square() + residual[..., 1].square()
                        ).sum(dim=-1).mean()
                        diag.downstream_delta_l2 = float(delta.item())
                        if prev_qlc_residual is not None:
                            r0 = prev_qlc_residual.reshape(-1, 2)
                            r1 = residual.reshape(-1, 2)
                            dot = (r0 * r1).sum()
                            n0 = r0.square().sum().sqrt()
                            n1 = r1.square().sum().sqrt()
                            diag.memory_echo_cos = float(
                                (dot / (n0 * n1).clamp_min(1e-12)).item()
                            )
                    qlc_diags.append(diag)

        if return_qlc_diag:
            self._last_qlc_diags = qlc_diags
            self._last_qlc_diag = qlc_diags[-1] if qlc_diags else None

        lm = self.lm_head_norm(self.lm_head_proj(z))
        logits = (
            lm[..., 0] @ self.embed.embed_real.weight.T
            + lm[..., 1] @ self.embed.embed_imag.weight.T
        )

        if self.has_qlc():
            aux_loss = self.qlc_cfg.ponder_lambda * ponder_cost
            # v8.2: pull the OrthoHalt target_mlp's u toward psi so |u^H psi|^2
            # rises off the 1/d noise floor and alpha/gamma actually carry
            # signal. (1 - align) is clamped at zero in case any path produces
            # an above-unit alignment via numerics.
            tw = float(self.qlc_cfg.target_alignment_weight)
            if tw > 0.0 and align_signals:
                align_mean = torch.stack(align_signals).mean()
                target = float(getattr(self.qlc_cfg, "target_alignment_target", 1.0))
                if target >= 1.0:
                    # v8.2 compatibility: force alignment upward. This fixed
                    # the 1/d floor but can saturate beta to zero.
                    align_loss = (1.0 - align_mean).clamp_min(0.0)
                else:
                    # V8.3 anti-collapse: keep beta headroom by aiming for a
                    # high-but-not-perfect target instead of align=1.
                    target_t = align_mean.new_tensor(target)
                    align_loss = (align_mean - target_t).square()
                aux_loss = aux_loss + tw * align_loss
        else:
            aux_loss = ponder_cost
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
        qlc_p = sum(
            p.numel()
            for module in self.iter_qlc_modules()
            for p in module.parameters()
        )
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

    def last_qlc_diagnostics_all(self) -> List[QLCDiagnostics]:
        return list(self._last_qlc_diags)


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
