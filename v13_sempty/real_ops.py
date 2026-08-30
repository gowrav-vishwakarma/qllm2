"""Fully-real PAM primitives — NamedTensor in, NamedTensor out.

The real twin of ``complex_ops.py``. Where the complex model packs every
feature as a split-real ``complex_pair`` (index 0 = real, index 1 = imag) and
runs four real GEMMs per projection, the real model keeps one real width and
does the identical recurrence in plain real arithmetic: a single GEMM, no
phase to manage, no conjugates in the write.

``model.py`` uses four things from here: the real RoPE table (the only raw
torch object in this module — built once, outside the graph, and declared in
``check_torch_layout.py``), ``RealEmbed`` (a single real embedding, tied to
the head), ``RealNorm`` (real RMSNorm that normalises the last feature axis
whatever the caller named it — the twin of ``ComplexNorm``), and
``RealGatedUnit`` (plain SwiGLU — the complex magnitude/phase gating has no
real analogue, so this is the standard real gated unit with the same
parameter names as the complex one).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sempyt.dim import Dim
from sempyt.nn import Linear as NamedLinear, RMSNorm
from sempyt.ops import at
from sempyt.structural import silu
from sempyt.tensor import NamedTensor, named


class RealEmbed(nn.Module):
    """A single real embedding, tied to the head.

    The score of a candidate token is the real dot product of its embedding
    row with the (real) hidden state — one matrix, one dot. (The complex
    head needed two, ``real . E_r + imag . E_i``; the real head needs one.)
    """

    def __init__(self, vocab_size: int, feature: Dim):
        super().__init__()
        self.dim = feature.size
        self.vocab = Dim("vocab", vocab_size)
        self.feature = feature
        self.embed = nn.Embedding(vocab_size, feature.size)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, ids: torch.Tensor, *id_layout: Dim) -> NamedTensor:
        table = named(self.embed.weight, (self.vocab, self.feature))
        return at(table, named(ids, id_layout))


class RealNorm(RMSNorm):
    """RMSNorm that normalises the last feature axis, whatever it is named.

    The real twin of ``ComplexNorm``. ``sempyt``'s ``RMSNorm`` matches its
    dim by identity, so a head whose feature is a distinct ``Dim`` of the
    same size (``lm_head_out``) would not be found; aliasing the last axis
    onto the stored ``feature`` makes "normalise the feature" layout-agnostic.
    """

    def __init__(self, feature: Dim, eps: float = 1e-6):
        super().__init__(feature, eps)
        self.feature = feature

    def forward(self, z: NamedTensor) -> NamedTensor:
        feature = [d for d in z.layout][-1]
        return super().forward(z.alias(feature, self.feature))


class RealGatedUnit(nn.Module):
    """Plain SwiGLU on a real width: ``down(silu(gate(x)) * up(x))``.

    Parameter names (``gate_proj`` / ``up_proj`` / ``down_proj``) mirror
    ``ComplexGatedUnit`` so the two models carry the same block structure.
    """

    def __init__(self, feature: Dim, expand: int = 3, activation: str = 'swish'):
        super().__init__()
        self.feature = feature
        self.hidden = Dim("hidden", feature.size * expand)
        self.gate_proj = NamedLinear(self.feature, self.hidden, bias=False)
        self.up_proj = NamedLinear(self.feature, self.hidden, bias=False)
        self.down_proj = NamedLinear(self.hidden, self.feature, bias=False)
        self.activation = activation

    def forward(self, x: NamedTensor) -> NamedTensor:
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


def build_rope_cache_real(max_len: int, n_pairs: int) -> torch.Tensor:
    """RoPE cos/sin table for a real head: one 2×2 rotation per channel pair.

    Pair ``i`` (channels ``2i``, ``2i+1``) rotates at frequency
    ``1 / 10000^(i / n_pairs)`` — the same frequencies as the complex head's
    ``n_pairs``-dim RoPE, which pairs channels ``2i``, ``2i+1``. Returns
    ``[max_len, n_pairs, 2]`` with the last axis holding ``[cos, sin]``.
    """
    inverse_freqs = 1.0 / (10000.0 ** (torch.arange(n_pairs).float() / n_pairs))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * inverse_freqs.unsqueeze(0)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)
