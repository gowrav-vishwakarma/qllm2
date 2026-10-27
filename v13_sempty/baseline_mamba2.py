"""Mamba-2 baseline on the v13_sempty training path (the "wind tunnel").

The reference Mamba-2 (state-spaces/mamba ``mamba_ssm`` 2.2.5: SSD Triton
scan, causal_conv1d, fused add+RMSNorm, its own init) behind the interface
``train.Trainer`` and the probes use from ``model.LM``:

  ``_hidden_to_lm`` / ``ce_from_lm``  the SAME fused chunked CE as PAM
                                      (answer-weight ``loss_mask`` included)
  ``forward`` -> (logits, None, aux)  probes / PG-19 eval
  ``generate``, ``count_parameters``, ``config`` (dataclass, saved in ckpts)

so data, loss, LR schedule, checkpoints, resume and every eval are shared
with PAM; the only variable is the sequence mixer. Raw torch by design (a
third-party reference model) -> listed in ``check_torch_layout.SKIP_FILES``.

Default size is param-matched to the PAM reference (102.1M): d_model 768,
17 layers, d_state 128, headdim 64 (24 heads), expand 2 -> 102.6M with the
50,261-token chat vocab (embedding padded to 50,264 rows; the head and the
softmax use only the real 50,261).

NOT a pip dep of the repo. Install on a box (cp311 / torch 2.8 / CUDA 12;
``--no-deps`` so torch/triton/transformers stay as they are):

  ~/.local/bin/uv pip install --python .venv/bin/python --no-deps \
    https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
    https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
    einops
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _transformers_compat() -> None:
    """mamba_ssm 2.2.5 imports two names transformers 5 dropped; both were
    deprecated aliases of ``GenerateDecoderOnlyOutput``. Restore the aliases
    before the first ``mamba_ssm`` import (no installed file is edited)."""
    import transformers.generation as tg
    for name in ('GreedySearchDecoderOnlyOutput', 'SampleDecoderOnlyOutput'):
        if not hasattr(tg, name):
            setattr(tg, name, tg.GenerateDecoderOnlyOutput)


@dataclass
class Mamba2Config:
    arch: str = 'mamba2'
    vocab_size: int = 50261
    d_model: int = 768
    n_layer: int = 17
    d_state: int = 128
    headdim: int = 64
    expand: int = 2
    d_conv: int = 4
    chunk_size: int = 256
    ngroups: int = 1


class Mamba2LM(nn.Module):
    """Reference ``MambaLMHeadModel`` (Mamba-2 mixer, tied head)."""

    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        _transformers_compat()
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.config = cfg
        self.net = MambaLMHeadModel(MambaConfig(
            d_model=cfg.d_model, n_layer=cfg.n_layer, vocab_size=cfg.vocab_size,
            ssm_cfg={'layer': 'Mamba2', 'd_state': cfg.d_state,
                     'headdim': cfg.headdim, 'expand': cfg.expand,
                     'd_conv': cfg.d_conv, 'chunk_size': cfg.chunk_size,
                     'ngroups': cfg.ngroups},
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
            pad_vocab_size_multiple=8, tie_embeddings=True,
        ))

    @property
    def blocks(self) -> nn.ModuleList:
        """The mixer layers (Trainer: per-block grad norms). A property, not
        an attribute, so the state_dict holds each weight once."""
        return self.net.backbone.layers

    def _head_weight(self) -> torch.Tensor:
        return self.net.lm_head.weight[: self.config.vocab_size]  # drop pad rows

    def _hidden_to_lm(self, input_ids, step_offset: int = 0):
        hidden = self.net.backbone(input_ids)  # [B, T, D], after the final norm
        return hidden, torch.zeros((), device=input_ids.device)

    def ce_from_lm(self, lm, labels, loss_mask=None, ignore_index=-100,
                   chunk: int = 4096, return_nll: bool = False, gemm_dtype=None):
        """The PAM loss: ``v13_sempty.fused_ce`` on (hidden, tied embedding)."""
        from v13_sempty.fused_ce import fused_linear_cross_entropy
        b, t, d = lm.shape
        out = fused_linear_cross_entropy(
            lm.reshape(b * t, d), self._head_weight(), labels.reshape(-1),
            mask=(loss_mask.reshape(-1) if loss_mask is not None else None),
            chunk=chunk, ignore_index=ignore_index, return_nll=return_nll,
            gemm_dtype=gemm_dtype,
        )
        if return_nll:
            return out, getattr(out, '_nll', None).reshape(b, t)
        return out

    def forward(self, input_ids, states=None, step_offset: int = 0, labels=None):
        hidden, aux = self._hidden_to_lm(input_ids)
        return F.linear(hidden, self._head_weight()), None, aux

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0,
                 top_k=50, top_p=0.0, repetition_penalty=1.0, eos_token_id=None):
        """Sampling for the in-training text samples (full re-forward per
        token: only ever ~80 tokens, so O(T^2) is irrelevant here)."""
        self.eval()
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)[0][:, -1].float()
            if repetition_penalty != 1.0:
                score = torch.gather(logits, 1, generated)
                score = torch.where(score > 0, score / repetition_penalty,
                                    score * repetition_penalty)
                logits.scatter_(1, generated, score)
            if temperature <= 0.0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = logits.topk(min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = float('-inf')
                if top_p > 0:
                    sl, si = logits.sort(descending=True)
                    p = sl.softmax(dim=-1)
                    sl[p.cumsum(dim=-1) - p >= top_p] = float('-inf')
                    logits = sl.scatter(1, si, sl)
                nxt = torch.multinomial(logits.softmax(dim=-1), 1)
            generated = torch.cat([generated, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
        return generated

    def count_parameters(self) -> Dict[str, int]:
        embed_p = self.net.backbone.embedding.weight.numel()
        block_p = sum(p.numel() for p in self.blocks.parameters())
        norm_p = sum(p.numel() for p in self.net.backbone.norm_f.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {'embedding (tied)': embed_p, 'blocks': block_p,
                'norms': norm_p, 'total': total}
