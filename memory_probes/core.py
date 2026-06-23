"""Shared complex matrix-memory math (NumPy reference implementation)."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch

ArrayC = np.ndarray  # complex128 [d] or [d,d]

STANDARD_DISTANCES = (64, 128, 256, 512, 1024, 2048)
LONG_CONTEXT_DISTANCES = (
    2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
)


def rand_unit_complex(rng: np.random.Generator, shape: Tuple[int, ...]) -> ArrayC:
    z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
    return z


def outer_v_kstar(v: ArrayC, k: ArrayC) -> ArrayC:
    return np.outer(v, np.conj(k))


def mat_vec(S: ArrayC, q: ArrayC) -> ArrayC:
    return S @ q


def inner(a: ArrayC, b: ArrayC) -> complex:
    return np.vdot(a, b)


def abs_inner(a: ArrayC, b: ArrayC) -> float:
    return float(np.abs(inner(a, b)))


def effective_rank(S: ArrayC) -> float:
    sv = np.linalg.svd(S, compute_uv=False)
    s = sv / (sv.sum() + 1e-12)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    return float(np.exp(-np.sum(s * np.log(s))))


def effective_rank_from_sv(sv: Sequence[float]) -> float:
    s = np.array(sv, dtype=np.float64)
    s = s / (s.sum() + 1e-12)
    s = s[s > 1e-12]
    return float(np.exp(-np.sum(s * np.log(s)))) if s.size else 0.0


def to_split_real(z: ArrayC) -> torch.Tensor:
    return torch.stack(
        [torch.from_numpy(z.real.astype(np.float64)),
         torch.from_numpy(z.imag.astype(np.float64))],
        dim=-1,
    )


def from_split_real(t: torch.Tensor) -> ArrayC:
    return t[..., 0].numpy() + 1j * t[..., 1].numpy()


def default_gamma_from_bias(bias: float = -4.0) -> float:
    dt = math.log1p(math.exp(bias))  # softplus
    return math.exp(-dt)


def gsp_gamma(dt: float, p: float) -> float:
    """GSP: gamma = exp(-dt)*(1-p) + p."""
    return math.exp(-dt) * (1.0 - p) + p


def long_context_distance_sweep(max_distance: int = 262144) -> Tuple[int, ...]:
    """Log-spaced distances from 2K up to max_distance (inclusive)."""
    out = [d for d in LONG_CONTEXT_DISTANCES if d <= max_distance]
    if not out or out[-1] < max_distance:
        if max_distance not in out:
            out.append(max_distance)
    return tuple(sorted(set(out)))


def default_dt() -> float:
    return math.log1p(math.exp(-4.0))


def pam_step_additive(
    S: ArrayC,
    g: float | np.ndarray,
    v: ArrayC,
    k: ArrayC,
) -> ArrayC:
    """S = g*S + v (x) k* ; g scalar or per-key-channel vector [d]."""
    if np.ndim(g) == 0:
        return g * S + outer_v_kstar(v, k)
    return S * g[np.newaxis, :] + outer_v_kstar(v, k)


def pam_step_delta(
    S: ArrayC,
    g: float,
    v: ArrayC,
    k: ArrayC,
    beta: float,
) -> ArrayC:
    S = g * S
    pred = S @ k
    u = beta * (v - pred)
    return S + outer_v_kstar(u, k)


def retrieve_score(S: ArrayC, k_query: ArrayC, v_target: ArrayC) -> float:
    y = mat_vec(S, k_query)
    return abs_inner(y, v_target)


def retrieval_accuracy(keys: ArrayC, values: ArrayC, S: ArrayC) -> float:
    n = keys.shape[0]
    correct = 0
    for i in range(n):
        y = mat_vec(S, keys[i])
        sims = np.array([abs_inner(y, values[j]) for j in range(n)])
        if int(np.argmax(sims)) == i:
            correct += 1
    return correct / n


def relative_retrieval(S: ArrayC, k: ArrayC, v: ArrayC) -> float:
    """Retrieval score normalized against a fresh single-write baseline."""
    baseline = retrieve_score(outer_v_kstar(v, k), k, v)
    return retrieve_score(S, k, v) / (baseline + 1e-12)


def simulate_filler_steps(
    S: ArrayC,
    T: int,
    rng: np.random.Generator,
    d: int,
    gamma: float,
    *,
    p_fill: float = 0.0,
    dt: float | None = None,
    report_every: int = 0,
) -> ArrayC:
    """Apply T haystack filler steps after an initial state S."""
    if T <= 0:
        return S
    if p_fill >= 1.0 - 1e-12:
        g = gsp_gamma(dt if dt is not None else default_dt(), p_fill)
        return (g ** T) * S
    g = gsp_gamma(dt, p_fill) if (dt is not None and p_fill > 0) else gamma
    v_scale = 1.0 - p_fill
    for t in range(1, T + 1):
        if report_every and t % report_every == 0:
            print(f'    ... filler step {t:,}/{T:,}', flush=True)
        k_f = rand_unit_complex(rng, (d,))
        v_f = rand_unit_complex(rng, (d,))
        S = pam_step_additive(S, g, v_f * v_scale, k_f)
    return S
