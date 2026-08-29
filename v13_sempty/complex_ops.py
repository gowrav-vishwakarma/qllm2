"""Split-real complex primitives — NamedTensor in, NamedTensor out.

``complex_pair`` is the last axis of size 2: index 0 = real, index 1 = imag.

The simple PAM model (``model.py``) uses only: the named linear/norm wrappers
(``ComplexLinear`` / ``ComplexNorm``), the activation fan (``ModReLU`` /
``ModSwish`` / ``PhaseModulatedActivation``), ``ComplexGatedUnit``,
``ComplexEmbed``, the shared dropout mask, and the RoPE table. The RoPE table
is the only raw-torch object in this module; it is built once, outside the
graph, and sliced by position (declared in ``check_torch_layout.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sempyt.dim import Dim
from sempyt.nn import ComplexLinear as _NamedComplexLinear
from sempyt.nn import ComplexRMSNorm as _NamedComplexRMSNorm
from sempyt.ops import as_complex, imag, real
from sempyt.policies import SplitComplex
from sempyt.structural import apply, cat, cos, ones, relu, sigmoid, sin
from sempyt.tensor import NamedTensor, named


def _feature_axis(z: NamedTensor) -> Dim:
    """The last non-pair axis — the feature, whatever the caller named it."""
    pair = z.policy.pair
    return [d for d in z.layout if d is not pair][-1]

def as_complex_dropout_mask(dropout_module: nn.Dropout, like: NamedTensor) -> NamedTensor:
    """Real dropout mask shared by both complex parts of ``like``."""
    lead = tuple(d for d in like.layout if d is not like.policy.pair)
    return apply(ones(*lead, device=like.device, dtype=like.dtype), dropout_module)


def to_real_concat(x: NamedTensor, into: Dim | None = None) -> NamedTensor:
    """``concat(real, imag)`` along the feature axis."""
    feature = _feature_axis(x)
    out = cat([real(x), imag(x)], over=feature)
    if into is not None:
        old = [d for d in out.layout if d.name == feature.name][0]
        out = out.alias(old, into)
    return out

class ComplexLinear(_NamedComplexLinear):
    """v13-compatible ComplexLinear. NamedTensor in, NamedTensor out."""

    def __init__(self, in_dim, out_dim, bias: bool = True, pair: Dim | None = None):
        in_feature = in_dim if isinstance(in_dim, Dim) else Dim("in_feature", in_dim)
        out_feature = out_dim if isinstance(out_dim, Dim) else Dim("out_feature", out_dim)
        complex_pair = pair if pair is not None else Dim("complex_pair", 2)
        super().__init__(in_feature, out_feature, complex_pair, bias)
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.complex_pair = complex_pair
        self.in_dim = in_feature.size
        self.out_dim = out_feature.size

    def forward(self, x: NamedTensor) -> NamedTensor:
        # v13 projects "the last feature axis", whatever the caller named it.
        return super().forward(x.alias(_feature_axis(x), self.in_feature))


class ComplexNorm(_NamedComplexRMSNorm):
    """v13 ComplexNorm (``scale`` param). NamedTensor in, NamedTensor out."""

    def __init__(self, dim, eps: float = 1e-6, pair: Dim | None = None):
        feature = dim if isinstance(dim, Dim) else Dim("feature", dim)
        complex_pair = pair if pair is not None else Dim("complex_pair", 2)
        super().__init__(feature, complex_pair, eps)
        self.feature = feature
        self.complex_pair = complex_pair

    def forward(self, z: NamedTensor) -> NamedTensor:
        # v13 normalises "the last feature axis", whatever the caller named it.
        return super().forward(z.alias(_feature_axis(z), self.feature))


class ModReLU(nn.Module):
    def __init__(self, feature: Dim, complex_pair: Dim):
        super().__init__()
        self.bias = nn.Parameter(torch.full((feature.size,), -0.1))
        self.feature = feature
        self.complex_pair = complex_pair
        self.policy = SplitComplex(complex_pair)

    def forward(self, z: NamedTensor) -> NamedTensor:
        feature = _feature_axis(z)
        magnitude = z.abs()
        activated = relu(magnitude + named(self.bias, (feature,)))
        return z * (activated / (magnitude + 1e-8))


class ModSwish(nn.Module):
    def __init__(self, feature: Dim, complex_pair: Dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(feature.size))
        self.beta = nn.Parameter(torch.ones(feature.size))
        self.feature = feature
        self.complex_pair = complex_pair
        self.policy = SplitComplex(complex_pair)

    def forward(self, z: NamedTensor) -> NamedTensor:
        feature = _feature_axis(z)
        magnitude = z.abs()
        gate = sigmoid(named(self.beta, (feature,)) * magnitude + named(self.bias, (feature,)))
        return z * ((magnitude * gate) / (magnitude + 1e-8))


class PhaseModulatedActivation(nn.Module):
    def __init__(self, feature: Dim, complex_pair: Dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(feature.size))
        self.beta = nn.Parameter(torch.ones(feature.size))
        self.phase_alpha = nn.Parameter(torch.zeros(feature.size))
        self.phase_beta = nn.Parameter(torch.zeros(feature.size))
        self.feature = feature
        self.complex_pair = complex_pair
        self.policy = SplitComplex(complex_pair)

    def forward(self, z: NamedTensor) -> NamedTensor:
        feature = _feature_axis(z)
        magnitude = z.abs()
        activated = magnitude * sigmoid(
            named(self.beta, (feature,)) * magnitude + named(self.bias, (feature,))
        )
        phase = z / (magnitude + 1e-8)
        angle = named(self.phase_alpha, (feature,)) * magnitude + named(self.phase_beta, (feature,))
        return (phase * as_complex(cos(angle), sin(angle), self.complex_pair)) * activated


def _build_activation(name: str, feature: Dim, complex_pair: Dim) -> nn.Module:
    if name == 'swish':
        return ModSwish(feature, complex_pair)
    if name == 'phase_mod':
        return PhaseModulatedActivation(feature, complex_pair)
    return ModReLU(feature, complex_pair)


class ComplexGatedUnit(nn.Module):
    """SwiGLU-style complex gating via named ComplexLinear + SplitComplex *."""

    def __init__(self, feature: Dim, expand: int = 3, activation: str = 'modrelu',
                 pair: Dim | None = None):
        super().__init__()
        self.feature = feature
        self.hidden = Dim("hidden", feature.size * expand)
        self.complex_pair = pair if pair is not None else Dim("complex_pair", 2)
        self.policy = SplitComplex(self.complex_pair)
        self.gate_proj = ComplexLinear(self.feature, self.hidden, bias=False, pair=self.complex_pair)
        self.up_proj = ComplexLinear(self.feature, self.hidden, bias=False, pair=self.complex_pair)
        self.down_proj = ComplexLinear(self.hidden, self.feature, bias=False, pair=self.complex_pair)
        self.act = _build_activation(activation, self.hidden, self.complex_pair)

    def forward(self, z: NamedTensor) -> NamedTensor:
        gate = self.gate_proj(z)
        up = self.act(self.up_proj(z))
        gate_magnitude = gate.abs()
        phase = gate / (gate_magnitude + 1e-8)
        return self.down_proj((phase * up) * sigmoid(gate_magnitude))


class ComplexEmbed(nn.Module):
    """Two real embeddings packed as SplitComplex via ``as_complex``."""

    def __init__(self, vocab_size: int, feature: Dim, complex_pair: Dim):
        super().__init__()
        self.dim = feature.size
        self.vocab = Dim("vocab", vocab_size)
        self.feature = feature
        self.complex_pair = complex_pair
        self.policy = SplitComplex(complex_pair)
        self.embed_real = nn.Embedding(vocab_size, feature.size)
        self.embed_imag = nn.Embedding(vocab_size, feature.size)
        nn.init.normal_(self.embed_real.weight, std=0.02)
        nn.init.normal_(self.embed_imag.weight, std=0.02)

    def forward(self, ids: torch.Tensor, *id_layout: Dim) -> NamedTensor:
        lead = tuple(id_layout)
        real_embed = named(self.embed_real(ids), lead + (self.feature,))
        imag_embed = named(self.embed_imag(ids), lead + (self.feature,))
        return as_complex(real_embed, imag_embed, self.complex_pair)


def build_rope_cache(max_len: int, head_dim: int) -> torch.Tensor:
    inverse_freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * inverse_freqs.unsqueeze(0)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)

