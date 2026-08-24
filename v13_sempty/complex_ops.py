"""Split-real complex primitives — NamedTensor in, NamedTensor out.

``complex_pair`` is the last axis of size 2: index 0 = real, index 1 = imag.

Parameter names match v13 (``weight_real``, ``scale``, …) so a v13
``state_dict`` loads without remapping.

Two raw-torch exits live here and nowhere else: ``fused_decay_matrix`` (the
``[time, time]`` lag table, which has no named form) and ``real_part`` /
``imag_part``, which feed the chunked-CE autograd Function.
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

REAL = 0
IMAG = 1


def _feature_axis(z: NamedTensor) -> Dim:
    pair = z.policy.pair
    return [d for d in z.layout if d is not pair][-1]


def real_part(z: torch.Tensor) -> torch.Tensor:
    """Raw exit for the chunked-CE Function, which packs concat(real, imag)."""
    return z[..., REAL]


def imag_part(z: torch.Tensor) -> torch.Tensor:
    """Raw exit for the chunked-CE Function, which packs concat(real, imag)."""
    return z[..., IMAG]


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


def fused_decay_matrix(decay_gamma: torch.Tensor, seq_len: int) -> torch.Tensor:
    """v13 ``_pt_decay_matrix``. No named form — this is a [time, time] kernel table."""
    log_gamma = torch.log(decay_gamma + 1e-6)
    cum_neg_log_gamma = torch.cumsum(-log_gamma, dim=-1)
    log_decay = (cum_neg_log_gamma.unsqueeze(-1) - cum_neg_log_gamma.unsqueeze(-2)).transpose(-1, -2)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=decay_gamma.device))
    log_decay = log_decay * causal + (1 - causal) * (-1e4)
    return torch.exp(log_decay.clamp(max=0.0))


def named_decay_matrix(gamma: NamedTensor, time: Dim, source_time: Dim) -> NamedTensor:
    """Named wrap of ``fused_decay_matrix``. ``gamma`` has ``time`` last."""
    g = gamma.to(*[d for d in gamma.layout if d is not time], time)
    lead = [d for d in g.layout if d is not time]
    flat = g.data.reshape(-1, time.size)
    decay = fused_decay_matrix(flat, time.size).reshape(*g.data.shape[:-1], time.size, source_time.size)
    return named(decay, tuple(lead) + (time, source_time))


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


class ComplexPosEmbed(nn.Module):
    def __init__(self, max_seq_len: int, feature: Dim):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.feature = feature
        self.pos_embed = nn.Embedding(max_seq_len, feature.size)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, z: NamedTensor, step_offset: int = 0, time: Dim | None = None) -> NamedTensor:
        time_axis = time or next(d for d in z.layout if d.name == "time")
        seq_len = z.size(time_axis)
        feature = next(
            d for d in z.layout if d.size == self.feature.size and d is not z.policy.pair
        )
        position_end = step_offset + seq_len
        if position_end > self.max_seq_len:
            raise ValueError(
                f"Position range [{step_offset}, {position_end}) exceeds max_seq_len "
                f"{self.max_seq_len}"
            )
        position_ids = torch.arange(step_offset, position_end, device=z.device)
        return z + named(self.pos_embed(position_ids), (time_axis, feature))


def build_rope_cache(max_len: int, head_dim: int) -> torch.Tensor:
    inverse_freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * inverse_freqs.unsqueeze(0)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)

