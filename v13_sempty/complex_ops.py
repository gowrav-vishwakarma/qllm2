"""Split-real complex primitives — NamedTensor-native, sempyt-backed.

Public torch ``[..., dim, 2]`` I/O is a thin wrap/unwrap so v13 selftests can
pass raw tensors. The math is ``contract`` / ``.abs()`` / SplitComplex ``*``,
not ``F.linear`` / ``[..., 0]`` bookkeeping.

Parameter names match v13 (``weight_real``, ``scale``, …) so a v13
``state_dict`` loads without remapping.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from sempyt.dim import Dim
from sempyt.nn import ComplexLinear as _NamedComplexLinear
from sempyt.nn import ComplexRMSNorm as _NamedComplexRMSNorm
from sempyt.ops import as_complex, imag, real
from sempyt.policies import SplitComplex
from sempyt.structural import cat, relu, sigmoid
from sempyt.tensor import NamedTensor, named

TensorLike = Union[torch.Tensor, NamedTensor]

REAL = 0
IMAG = 1


def _is_named(z) -> bool:
    return isinstance(z, NamedTensor)


def _wrap_last2(z: torch.Tensor, feat: Dim, px: Dim, policy: SplitComplex) -> NamedTensor:
    lead = z.shape[:-2]
    layout = [Dim(f"_b{i}", s) for i, s in enumerate(lead)] + [feat, px]
    return named(z, layout, policy)


def _maybe_wrap(z: TensorLike, feat: Dim, px: Dim, policy: SplitComplex) -> NamedTensor:
    if _is_named(z):
        return z
    return _wrap_last2(z, feat, px, policy)


def _maybe_unwrap(out: NamedTensor, was_named: bool) -> TensorLike:
    return out if was_named else out.data


def real_part(z: TensorLike) -> torch.Tensor:
    if _is_named(z):
        return real(z).data
    return z[..., REAL]


def imag_part(z: TensorLike) -> torch.Tensor:
    if _is_named(z):
        return imag(z).data
    return z[..., IMAG]


def stack_complex(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    return torch.stack([re, im], dim=-1)


def scale_complex(z: TensorLike, scale: TensorLike) -> TensorLike:
    """Multiply complex z by a real scale (named broadcast, or torch fallback)."""
    if _is_named(z) and _is_named(scale):
        return z * scale
    if _is_named(z):
        return z * scale if not torch.is_tensor(scale) else z * float(scale) if scale.ndim == 0 else _scale_raw(z.data, scale)
    z_t = z.data if _is_named(z) else z
    sc = scale.data if _is_named(scale) else scale
    return _scale_raw(z_t, sc)


def _scale_raw(z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    while scale.dim() < z.dim():
        scale = scale.unsqueeze(-1)
    return z * scale


def as_complex_dropout_mask(dropout_module: nn.Dropout, like_z: TensorLike) -> torch.Tensor:
    data = like_z.data if _is_named(like_z) else like_z
    return dropout_module(torch.ones(data.shape[:-1], device=data.device, dtype=data.dtype))


_PX = Dim("px", 2)
_SPLIT = SplitComplex(_PX)


def cmul(a: TensorLike, b: TensorLike) -> TensorLike:
    """SplitComplex multiply. Named if both inputs are named."""
    if _is_named(a) and _is_named(b):
        return a * b
    return _SPLIT.mul(
        a.data if _is_named(a) else a,
        b.data if _is_named(b) else b,
    )


def cconj(x: TensorLike) -> TensorLike:
    if _is_named(x):
        return x.conj()
    return _SPLIT.conj(x)


def cabs(x: TensorLike) -> TensorLike:
    if _is_named(x):
        return x.abs()
    return _SPLIT.abs(x)


def cnormalize(x: TensorLike) -> TensorLike:
    """Per-element unit magnitude (phase preserved)."""
    if _is_named(x):
        return x / x.abs()
    px = Dim("px", 2)
    feat = Dim("feat", x.shape[-2])
    xn = _wrap_last2(x, feat, px, SplitComplex(px))
    return (xn / xn.abs()).data


def cnormalize_vec(x: TensorLike) -> TensorLike:
    """Per-VECTOR unit norm across the feature dim (v13 ``cnormalize_vec``)."""
    if _is_named(x):
        pair = x.policy.pair
        feat = [d for d in x.layout if d is not pair][-1]
        return x.normalize(over=feat)
    px = Dim("px", 2)
    feat = Dim("feat", x.shape[-2])
    return _wrap_last2(x, feat, px, SplitComplex(px)).normalize(over=feat).data


def to_real_concat(x: TensorLike, into: Dim | None = None) -> TensorLike:
    """``concat(real, imag)`` along the feature axis."""
    if _is_named(x):
        pair = x.policy.pair
        feat = [d for d in x.layout if d is not pair][-1]
        out = cat([real(x), imag(x)], over=feat)
        if into is not None:
            old = [d for d in out.layout if d.name == feat.name][0]
            out = out.alias(old, into)
        return out
    return torch.cat([x[..., 0], x[..., 1]], dim=-1)


def fused_decay_matrix(decay_gamma: torch.Tensor, seq_len: int) -> torch.Tensor:
    """v13 ``_pt_decay_matrix``. No named form — this is a [T,T] kernel table."""
    log_gamma = torch.log(decay_gamma + 1e-6)
    cum_neg_log_gamma = torch.cumsum(-log_gamma, dim=-1)
    log_decay = (cum_neg_log_gamma.unsqueeze(-1) - cum_neg_log_gamma.unsqueeze(-2)).transpose(-1, -2)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=decay_gamma.device))
    log_decay = log_decay * causal + (1 - causal) * (-1e4)
    return torch.exp(log_decay.clamp(max=0.0))


def named_decay_matrix(gamma: NamedTensor, time: Dim, src: Dim) -> NamedTensor:
    """Named wrap of ``fused_decay_matrix``. ``gamma`` has ``time`` last."""
    g = gamma.to(*[d for d in gamma.layout if d is not time], time)
    lead = [d for d in g.layout if d is not time]
    flat = g.data.reshape(-1, time.size)
    D = fused_decay_matrix(flat, time.size).reshape(*g.data.shape[:-1], time.size, src.size)
    return named(D, tuple(lead) + (time, src))


class ComplexLinear(_NamedComplexLinear):
    """v13-compatible ComplexLinear. Accepts ints or Dims; torch or NamedTensor."""

    def __init__(self, in_dim, out_dim, bias: bool = True, pair: Dim | None = None):
        inn = in_dim if isinstance(in_dim, Dim) else Dim("inn", in_dim)
        out = out_dim if isinstance(out_dim, Dim) else Dim("out", out_dim)
        px = pair if pair is not None else Dim("px", 2)
        super().__init__(inn, out, px, bias)
        self.in_dim = inn.size
        self.out_dim = out.size

    def forward(self, x: TensorLike) -> TensorLike:
        was = _is_named(x)
        if was:
            lead = [d for d in x.layout if d is not getattr(x.policy, "pair", None)][:-1]
            xn = named(x.data, tuple(lead) + (self.inn, self.pair), self.policy)
        else:
            xn = _wrap_last2(x, self.inn, self.pair, self.policy)
        out = super().forward(xn)
        return _maybe_unwrap(out, was)


class ComplexNorm(_NamedComplexRMSNorm):
    """v13 ComplexNorm (``scale`` param). Named I/O with torch wrap/unwrap."""

    def __init__(self, dim, eps: float = 1e-6, pair: Dim | None = None):
        feat = dim if isinstance(dim, Dim) else Dim("feat", dim)
        px = pair if pair is not None else Dim("px", 2)
        super().__init__(feat, px, eps)

    def forward(self, z: TensorLike) -> TensorLike:
        was = _is_named(z)
        zn = z if was else _wrap_last2(z, self.dim, self.pair, SplitComplex(self.pair))
        out = super().forward(zn)
        return _maybe_unwrap(out, was)


class ModReLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), -0.1))
        self.feat = Dim("feat", dim)
        self.px = Dim("px", 2)
        self.policy = SplitComplex(self.px)

    def forward(self, z: TensorLike) -> TensorLike:
        was = _is_named(z)
        zn = _maybe_wrap(z, self.feat, self.px, self.policy)
        feat = [d for d in zn.layout if d is not zn.policy.pair][-1]
        mag = zn.abs()
        activated = relu(mag + named(self.bias, (feat,)))
        out = zn * (activated / (mag + 1e-8))
        return _maybe_unwrap(out, was)


class ModSwish(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))
        self.feat = Dim("feat", dim)
        self.px = Dim("px", 2)
        self.policy = SplitComplex(self.px)

    def forward(self, z: TensorLike) -> TensorLike:
        was = _is_named(z)
        zn = _maybe_wrap(z, self.feat, self.px, self.policy)
        feat = [d for d in zn.layout if d is not zn.policy.pair][-1]
        mag = zn.abs()
        gate = sigmoid(named(self.beta, (feat,)) * mag + named(self.bias, (feat,)))
        activated = mag * gate
        out = zn * (activated / (mag + 1e-8))
        return _maybe_unwrap(out, was)


class PhaseModulatedActivation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))
        self.phase_alpha = nn.Parameter(torch.zeros(dim))
        self.phase_beta = nn.Parameter(torch.zeros(dim))
        self.feat = Dim("feat", dim)
        self.px = Dim("px", 2)
        self.policy = SplitComplex(self.px)

    def forward(self, z: TensorLike) -> TensorLike:
        was = _is_named(z)
        zn = _maybe_wrap(z, self.feat, self.px, self.policy)
        feat = [d for d in zn.layout if d is not zn.policy.pair][-1]
        mag = zn.abs()
        activated = mag * sigmoid(
            named(self.beta, (feat,)) * mag + named(self.bias, (feat,))
        )
        phase = zn / (mag + 1e-8)
        angle = named(self.phase_alpha, (feat,)) * mag + named(self.phase_beta, (feat,))
        from sempyt.structural import cos, sin
        rot = as_complex(cos(angle), sin(angle), self.px)
        out = (phase * rot) * activated
        return _maybe_unwrap(out, was)


def _build_activation(name: str, dim: int) -> nn.Module:
    if name == 'swish':
        return ModSwish(dim)
    if name == 'phase_mod':
        return PhaseModulatedActivation(dim)
    return ModReLU(dim)


class ComplexGatedUnit(nn.Module):
    """SwiGLU-style complex gating via named ComplexLinear + SplitComplex *."""

    def __init__(self, dim: int, expand: int = 3, activation: str = 'modrelu'):
        super().__init__()
        hidden_dim = dim * expand
        self.feat = Dim("feat", dim)
        self.hid = Dim("hid", hidden_dim)
        self.px = Dim("px", 2)
        self.policy = SplitComplex(self.px)
        self.gate_proj = ComplexLinear(self.feat, self.hid, bias=False, pair=self.px)
        self.up_proj = ComplexLinear(self.feat, self.hid, bias=False, pair=self.px)
        self.down_proj = ComplexLinear(self.hid, self.feat, bias=False, pair=self.px)
        self.act = _build_activation(activation, hidden_dim)

    def forward(self, z: TensorLike) -> TensorLike:
        was = _is_named(z)
        if was:
            lead = [d for d in z.layout if d is not z.policy.pair][:-1]
            zn = named(z.data, tuple(lead) + (self.feat, self.px), self.policy)
        else:
            zn = _maybe_wrap(z, self.feat, self.px, self.policy)
        gate = self.gate_proj(zn)
        up = self.act(self.up_proj(zn))
        if not _is_named(gate):
            gate = _maybe_wrap(gate, self.hid, self.px, self.policy)
        if not _is_named(up):
            up = _maybe_wrap(up, self.hid, self.px, self.policy)
        gmag = gate.abs()
        phase = gate / (gmag + 1e-8)
        gated = (phase * up) * sigmoid(gmag)
        out = self.down_proj(gated)
        return _maybe_unwrap(out if _is_named(out) else _maybe_wrap(out, self.feat, self.px, self.policy), was)


class ComplexEmbed(nn.Module):
    """Two real embeddings packed as SplitComplex via ``as_complex``."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.dim = dim
        self.vocab = Dim("vocab", vocab_size)
        self.feat = Dim("feat", dim)
        self.px = Dim("px", 2)
        self.policy = SplitComplex(self.px)
        self.embed_real = nn.Embedding(vocab_size, dim)
        self.embed_imag = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed_real.weight, std=0.02)
        nn.init.normal_(self.embed_imag.weight, std=0.02)

    def forward(self, ids: torch.Tensor, *id_layout: Dim) -> TensorLike:
        lead = tuple(id_layout) if id_layout else tuple(
            Dim(f"id{i}", s) for i, s in enumerate(ids.shape)
        )
        r = named(self.embed_real(ids), lead + (self.feat,))
        i = named(self.embed_imag(ids), lead + (self.feat,))
        out = as_complex(r, i, self.px)
        return out if id_layout else out.data


class ComplexPosEmbed(nn.Module):
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.Tmax = Dim("Tmax", max_seq_len)
        self.feat = Dim("feat", dim)
        self.px = Dim("px", 2)
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, z: TensorLike, step_offset: int = 0, time: Dim | None = None) -> TensorLike:
        was = _is_named(z)
        if was:
            T = time or next(d for d in z.layout if d.name in ("T", "time") or d is time)
            seq_len = z.size(T)
            feat = self.feat
            # rebind feat to z's feature dim identity if present
            feat = next((d for d in z.layout if d.size == self.feat.size and d is not z.policy.pair), self.feat)
        else:
            seq_len = z.shape[1]
            feat = self.feat
        position_end = step_offset + seq_len
        if position_end > self.max_seq_len:
            raise ValueError(
                f"Position range [{step_offset}, {position_end}) exceeds max_seq_len "
                f"{self.max_seq_len}"
            )
        position_ids = torch.arange(step_offset, position_end, device=(z.data if was else z).device)
        pe = named(self.pos_embed(position_ids), ((time or Dim("T", seq_len)), feat))
        if was:
            return z + pe
        return z + pe.data.unsqueeze(0).unsqueeze(-1)


def build_rope_cache(max_len: int, head_dim: int) -> torch.Tensor:
    inverse_freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * inverse_freqs.unsqueeze(0)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rope(q: TensorLike, k: TensorLike, rope: torch.Tensor) -> tuple[TensorLike, TensorLike]:
    """SplitComplex ``*`` of Q/K by a ``[T, d, 2]`` cache."""
    if _is_named(q):
        T = next(d for d in q.layout if d.size == rope.shape[0])
        d = next(d for d in q.layout if d.size == rope.shape[1] and d is not q.policy.pair)
        rn = named(rope.to(dtype=q.dtype, device=q.device), (T, d, q.policy.pair), q.policy)
        return q * rn, k * rn
    T = Dim("T", q.shape[-3])
    d = Dim("d", q.shape[-2])
    px = Dim("px", 2)
    pol = SplitComplex(px)
    lead = [Dim(f"q{i}", s) for i, s in enumerate(q.shape[:-3])]
    qn = named(q, lead + [T, d, px], pol)
    kn = named(k, [Dim(f"k{i}", s) for i, s in enumerate(k.shape[:-3])] + [T, d, px], pol)
    rn = named(rope.to(dtype=q.dtype, device=q.device), (T, d, px), pol)
    return (qn * rn).data, (kn * rn).data
