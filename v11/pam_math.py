"""V11 PAM math probes — no trained checkpoint required.

Pure NumPy/PyTorch experiments for binding capacity, persistence, interference,
state rank, conjugate retrieval, layer-bridge correctness, and needle-in-haystack.

Run:
    .venv/bin/python -m v11.pam_math --all
    .venv/bin/python -m v11.pam_math --test binding
    .venv/bin/python -m v11.pam_math --test persistence --distances 64,128,512,2048
    .venv/bin/python -m v11.pam_math --test niah --distances 64,128,512,1024,2048
    .venv/bin/python -m v11.pam_math --test long-context
    .venv/bin/python -m v11.pam_math --test long-context --max-distance 1048576
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from v11.model import V11Config, V11PAMLayer

# ── Complex NumPy helpers (split-real convention: [..., 2] = re, im) ─────────

ArrayC = np.ndarray  # complex128 [d] or [d,d]


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


# Default distance sweeps (training seq_len=2048; long-context extends far beyond)
STANDARD_DISTANCES = (64, 128, 256, 512, 1024, 2048)
LONG_CONTEXT_DISTANCES = (
    2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
)


def long_context_distance_sweep(max_distance: int = 262144) -> Tuple[int, ...]:
    """Log-spaced distances from 2K up to max_distance (inclusive)."""
    out = [d for d in LONG_CONTEXT_DISTANCES if d <= max_distance]
    if not out or out[-1] < max_distance:
        if max_distance not in out:
            out.append(max_distance)
    return tuple(sorted(set(out)))


def default_dt() -> float:
    return math.log1p(math.exp(-4.0))


# ── Matrix PAM recurrence (NumPy reference) ──────────────────────────────────

def pam_step_additive(
    S: ArrayC,
    g: float | np.ndarray,
    v: ArrayC,
    k: ArrayC,
) -> ArrayC:
    """S = g*S + v (x) k* ; g scalar or per-key-channel vector [d]."""
    if np.ndim(g) == 0:
        return g * S + outer_v_kstar(v, k)
    # per-channel decay on key axis (columns)
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
    """Apply T haystack filler steps after an initial state S.

    Fast path: p_fill=1 → no value writes, pure decay S *= g^T.
    """
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


# ── Vector holographic (HRR-style) baseline ──────────────────────────────────

def hrr_bind(a: ArrayC, b: ArrayC) -> ArrayC:
    """Circular convolution binding in complex domain (FFT)."""
    fa, fb = np.fft.fft(a), np.fft.fft(b)
    return np.fft.ifft(fa * fb)


def hrr_unbind(state: ArrayC, key: ArrayC) -> ArrayC:
    fk = np.fft.fft(key)
    return np.fft.ifft(np.fft.fft(state) * np.conj(fk))


def hrr_binding_capacity(
    rng: np.random.Generator,
    d: int,
    ns: Sequence[int],
    trials: int,
) -> Dict[str, Any]:
    accs: List[float] = []
    for n in ns:
        trial_accs = []
        for _ in range(trials):
            keys = rand_unit_complex(rng, (n, d))
            values = rand_unit_complex(rng, (n, d))
            state = np.zeros(d, dtype=np.complex128)
            for i in range(n):
                state = state + hrr_bind(keys[i], values[i])
            correct = 0
            for i in range(n):
                retrieved = hrr_unbind(state, keys[i])
                sims = np.array([abs_inner(retrieved, values[j]) for j in range(n)])
                if int(np.argmax(sims)) == i:
                    correct += 1
            trial_accs.append(correct / n)
        accs.append(float(np.mean(trial_accs)))
    theory = [1.0 / math.sqrt(max(n, 1)) for n in ns]
    return {
        'ns': list(ns),
        'accuracy': accs,
        'theory_1_sqrt_n': theory,
        'accuracy_at_n1': accs[0] if accs else None,
        'accuracy_at_n_max': accs[-1] if accs else None,
    }


# ── A2: Binding capacity ─────────────────────────────────────────────────────

def matrix_binding_capacity(
    rng: np.random.Generator,
    d: int,
    ns: Sequence[int],
    trials: int,
) -> Dict[str, Any]:
    accs: List[float] = []
    for n in ns:
        trial_accs = []
        for _ in range(trials):
            keys = rand_unit_complex(rng, (n, d))
            values = rand_unit_complex(rng, (n, d))
            S = np.zeros((d, d), dtype=np.complex128)
            for i in range(n):
                S += outer_v_kstar(values[i], keys[i])
            trial_accs.append(retrieval_accuracy(keys, values, S))
        accs.append(float(np.mean(trial_accs)))
    return {
        'ns': list(ns),
        'accuracy': accs,
        'accuracy_at_n1': accs[0] if accs else None,
        'accuracy_at_n64': accs[ns.index(64)] if 64 in ns else None,
    }


def test_binding(
    d: int = 64,
    max_n: int = 200,
    trials: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    ns = list(range(1, 65, 4)) + [64] + list(range(65, max_n + 1, 8))
    ns = sorted(set(ns))
    matrix = matrix_binding_capacity(rng, d, ns, trials)
    vector = hrr_binding_capacity(rng, d, ns, trials)
    print('\n[A2] Binding capacity (matrix PAM vs vector HRR)')
    print(f'  d={d}, trials={trials}')
    for i, n in enumerate(ns):
        if n in (1, 16, 64, 128, max_n) or i == len(ns) - 1:
            print(f'  N={n:4d}  matrix={matrix["accuracy"][i]:.3f}  '
                  f'vector={vector["accuracy"][i]:.3f}  theory={vector["theory_1_sqrt_n"][i]:.3f}')
    return {'matrix_pam': matrix, 'vector_hrr': vector, 'd': d, 'trials': trials}


# ── A3: Persistence vs distance ──────────────────────────────────────────────

def test_persistence(
    d: int = 64,
    distances: Sequence[int] = (64, 128, 256, 512, 1024, 2048),
    gammas: Sequence[float] = (0.99, 0.995, 0.999, 1.0),
    filler_writes: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    k_needle = rand_unit_complex(rng, (d,))
    v_needle = rand_unit_complex(rng, (d,))

    rows: List[Dict[str, Any]] = []
    print('\n[A3] Association persistence vs distance')
    print(f'  d={d}, filler_writes/token={filler_writes}')
    for gamma in gammas:
        for T in distances:
            S = outer_v_kstar(v_needle, k_needle)
            for _ in range(T):
                for _w in range(filler_writes):
                    k_f = rand_unit_complex(rng, (d,))
                    v_f = rand_unit_complex(rng, (d,))
                    S = pam_step_additive(S, gamma, v_f, k_f)
            score = retrieve_score(S, k_needle, v_needle)
            score0 = retrieve_score(S * 0 + outer_v_kstar(v_needle, k_needle), k_needle, v_needle)
            rel = score / (score0 + 1e-12)
            rows.append({'gamma': gamma, 'distance': T, 'score': score, 'relative': rel})
        sub = [r for r in rows if r['gamma'] == gamma]
        print(f'  gamma={gamma:.4f}: ' +
              ', '.join(f'T={r["distance"]}:{r["relative"]:.3f}' for r in sub[:4]) +
              (' ...' if len(sub) > 4 else ''))

    return {'d': d, 'filler_writes': filler_writes, 'results': rows}


# ── A4: Multi-association interference ───────────────────────────────────────

def test_interference(
    d: int = 64,
    pair_counts: Sequence[int] = (4, 8, 16, 32, 64),
    filler_counts: Sequence[int] = (0, 64, 256, 1024),
    seed: int = 42,
    use_delta: bool = False,
    n_states: int = 1,
    state_dt_spread: float = 2.0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    print(f'\n[A4] Multi-association interference (delta={use_delta}, n_states={n_states})')

    for n_pairs in pair_counts:
        keys = rand_unit_complex(rng, (n_pairs, d))
        values = rand_unit_complex(rng, (n_pairs, d))
        for n_filler in filler_counts:
            if n_states == 1:
                S = np.zeros((d, d), dtype=np.complex128)
                gamma = 0.995
                beta = 0.5
                for i in range(n_pairs):
                    if use_delta:
                        S = pam_step_delta(S, gamma, values[i], keys[i], beta)
                    else:
                        S = pam_step_additive(S, gamma, values[i], keys[i])
                for _ in range(n_filler):
                    k_f = rand_unit_complex(rng, (d,))
                    v_f = rand_unit_complex(rng, (d,))
                    if use_delta:
                        S = pam_step_delta(S, gamma, v_f, k_f, beta)
                    else:
                        S = pam_step_additive(S, gamma, v_f, k_f)
                score = retrieve_score(S, keys[0], values[0])
            else:
                # E3: K states with different decay biases
                K = n_states
                offsets = np.linspace(-state_dt_spread, state_dt_spread, K)
                gammas = [default_gamma_from_bias(-4.0 + off) for off in offsets]
                states = [np.zeros((d, d), dtype=np.complex128) for _ in range(K)]
                phases = rng.uniform(0, 2 * np.pi, K)
                for i in range(n_pairs):
                    for s_idx in range(K):
                        states[s_idx] = pam_step_additive(
                            states[s_idx], gammas[s_idx], values[i], keys[i])
                for _ in range(n_filler):
                    k_f = rand_unit_complex(rng, (d,))
                    v_f = rand_unit_complex(rng, (d,))
                    for s_idx in range(K):
                        states[s_idx] = pam_step_additive(
                            states[s_idx], gammas[s_idx], v_f, k_f)
                y = np.zeros(d, dtype=np.complex128)
                for s_idx in range(K):
                    rot = np.exp(1j * phases[s_idx])
                    y += rot * (states[s_idx] @ keys[0])
                score = abs_inner(y, values[0])

            rows.append({
                'n_pairs': n_pairs,
                'n_filler': n_filler,
                'score': score,
                'use_delta': use_delta,
                'n_states': n_states,
            })
            if n_filler in (0, 256) and n_pairs in (8, 32):
                print(f'  pairs={n_pairs:2d} filler={n_filler:4d} score={score:.4f}')

    return {'d': d, 'results': rows}


# ── A5: State rank evolution ─────────────────────────────────────────────────

def test_rank(
    d: int = 64,
    steps: int = 512,
    gamma: float = 0.995,
    mode: str = 'random',
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    S = np.zeros((d, d), dtype=np.complex128)
    ranks: List[float] = []
    print(f'\n[A5] State rank evolution (mode={mode}, gamma={gamma}, steps={steps})')

    if mode == 'random':
        for t in range(steps):
            k = rand_unit_complex(rng, (d,))
            v = rand_unit_complex(rng, (d,))
            S = pam_step_additive(S, gamma, v, k)
            ranks.append(effective_rank(S))
    elif mode == 'overwrite':
        k = rand_unit_complex(rng, (d,))
        for t in range(steps):
            v = rand_unit_complex(rng, (d,))
            S = pam_step_additive(S, gamma, v, k)
            ranks.append(effective_rank(S))
    else:
        raise ValueError(f'unknown mode: {mode}')

    print(f'  rank@1={ranks[0]:.2f}  max={max(ranks):.2f}  final={ranks[-1]:.2f}  (d={d})')
    return {
        'd': d,
        'steps': steps,
        'gamma': gamma,
        'mode': mode,
        'ranks': ranks,
        'final_rank': ranks[-1],
        'max_rank': max(ranks),
    }


# ── A6: Conjugate retrieval interference ─────────────────────────────────────

def test_conjugate_interference(
    d: int = 64,
    T: int = 128,
    seed: int = 42,
    layer_bridge: bool = True,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    Q = rand_unit_complex(rng, (T, d))
    K = rand_unit_complex(rng, (T, d))

    # W[i,j] = K_j* · Q_i  →  [T,T]
    W = Q @ K.conj().T
    mask = np.tril(np.ones((T, T), dtype=bool), k=-1)
    w_real = W.real[mask]
    w_imag = W.imag[mask]
    frac_neg = float(np.mean(w_real < 0))

    out: Dict[str, Any] = {
        'd': d,
        'T': T,
        'frac_destructive': frac_neg,
        'mean_re': float(np.mean(w_real)),
        'std_re': float(np.std(w_real)),
        'mean_im': float(np.mean(w_imag)),
    }
    print('\n[A6] Conjugate retrieval interference (K*·Q)')
    print(f'  T={T}, d={d}, frac_destructive={frac_neg:.3f}, mean_Re={out["mean_re"]:.4f}')

    if layer_bridge:
        cfg = V11Config(
            dim=32, n_heads=2, head_dim=16, n_layers=1, expand=2,
            dropout=0.0, chunk_size=24, use_gsp=False, use_rope=False,
        )
        layer = V11PAMLayer(cfg, layer_idx=0).eval()
        B, T_small = 1, min(T, 32)
        x = torch.randn(B, T_small, cfg.dim, 2) * 0.3
        with torch.no_grad():
            q, k, v = layer._project(x, 0)
        h = 0
        Qr, Qi = q[0, h, :, :, 0].numpy(), q[0, h, :, :, 1].numpy()
        Kr, Ki = k[0, h, :, :, 0].numpy(), k[0, h, :, :, 1].numpy()
        W_real = Kr @ Qr.T + Ki @ Qi.T
        W_imag = Kr @ Qi.T - Ki @ Qr.T
        W_layer = W_real + 1j * W_imag
        mask_s = np.tril(np.ones((T_small, T_small), dtype=bool), k=-1)
        frac_layer = float(np.mean(W_layer.real[mask_s] < 0))
        out['layer_frac_destructive'] = frac_layer
        out['layer_T'] = T_small
        print(f'  V11PAMLayer (untrained): T={T_small}, frac_destructive={frac_layer:.3f}')

    return out


# ── A7: Layer bridge (NumPy vs PyTorch recurrent steps) ──────────────────────

def _numpy_recur_additive(S_np, g, v_np, k_np, q_np):
    S = S_np.copy()
    S = pam_step_additive(S, g, v_np, k_np)
    y = mat_vec(S, q_np)
    return y, S


def _torch_from_c(z: ArrayC, device='cpu') -> torch.Tensor:
    t = torch.zeros(*z.shape, 2, dtype=torch.float64, device=device)
    t[..., 0] = torch.from_numpy(z.real)
    t[..., 1] = torch.from_numpy(z.imag)
    return t


def test_layer_bridge(
    modes: Sequence[str] = ('baseline', 'e1', 'e2', 'e3'),
    d: int = 16,
    T: int = 40,
    seed: int = 0,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    print('\n[A7] V11PAMLayer synthetic bridge (NumPy vs PyTorch recurrent)')
    results: Dict[str, Any] = {}

    mode_cfgs = {
        'baseline': dict(decay_mode='head', write_mode='additive', n_states=1),
        'e1': dict(decay_mode='per_channel', write_mode='additive', n_states=1),
        'e2': dict(decay_mode='head', write_mode='delta', n_states=1, delta_chunk=20),
        'e3': dict(decay_mode='head', write_mode='additive', n_states=2, state_dt_spread=2.0),
    }

    for mode in modes:
        if mode not in mode_cfgs:
            continue
        torch.manual_seed(seed)
        cfg_kw = dict(
            vocab_size=512, dim=32, n_heads=2, head_dim=d, n_layers=1,
            expand=2, dropout=0.0, max_seq_len=512, chunk_size=24,
            gradient_checkpointing=False, use_rope=False, use_gsp=False,
        )
        cfg_kw.update(mode_cfgs[mode])
        cfg = V11Config(**cfg_kw)
        layer = V11PAMLayer(cfg, layer_idx=0).double().eval()

        rng = np.random.default_rng(seed)
        x = torch.randn(1, T, cfg.dim, 2, dtype=torch.float64) * 0.4

        with torch.no_grad():
            y_par, _ = layer(x, state=None, step_offset=0)
            y_steps = []
            state = None
            for t in range(T):
                y_t, state = layer(x[:, t:t + 1], state=state, step_offset=t)
                y_steps.append(y_t)
            y_rec = torch.cat(y_steps, dim=1)

        diff = (y_par - y_rec).abs().max().item()
        ok = diff < atol
        results[mode] = {'max_diff': diff, 'pass': ok}
        print(f'  [{mode:8s}] parallel vs recurrent max|Δ|={diff:.2e}  {"PASS" if ok else "FAIL"}')

        # Step-level additive bridge (baseline only)
        if mode == 'baseline':
            g = 0.995
            v_np = rand_unit_complex(rng, (d,))
            k_np = rand_unit_complex(rng, (d,))
            q_np = rand_unit_complex(rng, (d,))
            S0 = np.zeros((d, d), dtype=np.complex128)
            _, S_np = _numpy_recur_additive(S0, g, v_np, k_np, q_np)

            S_pt = torch.zeros(1, cfg.n_heads, d, d, 2, dtype=torch.float64)
            g_pt = torch.full((1, cfg.n_heads), g, dtype=torch.float64)
            v_pt = _torch_from_c(v_np).unsqueeze(0).unsqueeze(0)
            k_pt = _torch_from_c(k_np).unsqueeze(0).unsqueeze(0)
            q_pt = _torch_from_c(q_np).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                _, S_pt_out = layer._recur_step_additive(S_pt, g_pt, v_pt, k_pt, q_pt)
            S_pt_c = S_pt_out[0, 0, ..., 0].numpy() + 1j * S_pt_out[0, 0, ..., 1].numpy()
            step_diff = np.max(np.abs(S_np - S_pt_c))
            results['baseline_step_diff'] = float(step_diff)
            print(f'  [baseline] NumPy vs _recur_step_additive max|ΔS|={step_diff:.2e}')

    return results


# ── A8: Needle-in-haystack (math level) ──────────────────────────────────────

def niah_bare_decay(
    d: int,
    distances: Sequence[int],
    gammas: Sequence[float],
    seed: int = 42,
    report_every: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    k_n = rand_unit_complex(rng, (d,))
    v_n = rand_unit_complex(rng, (d,))
    baseline_S = outer_v_kstar(v_n, k_n)
    rows = []
    print('\n[A8.1] NIAH bare decay envelope')
    for gamma in gammas:
        for T in distances:
            S = baseline_S.copy()
            S = simulate_filler_steps(
                S, T, rng, d, gamma, report_every=report_every,
            )
            rel = relative_retrieval(S, k_n, v_n)
            theory = gamma ** T
            rows.append({
                'gamma': gamma, 'distance': T, 'relative': rel,
                'theory_gamma_T': theory, 'decay_only_relative': theory,
            })
        # Print a sample of distances (include extremes)
        sub = [r for r in rows if r['gamma'] == gamma]
        picks = {sub[0]['distance'], sub[len(sub) // 2]['distance'], sub[-1]['distance']}
        sub = [r for r in sub if r['distance'] in picks]
        if sub:
            print(f'  gamma={gamma:.4f}: ' +
                  ', '.join(f'T={r["distance"]:>7,}:rel={r["relative"]:.4f}' for r in sub))
    return {'results': rows}


def niah_gsp_sweep(
    d: int,
    distances: Sequence[int],
    protect_values: Sequence[float],
    dt: float | None = None,
    seed: int = 42,
    report_every: int = 0,
) -> Dict[str, Any]:
    if dt is None:
        dt = default_dt()
    rng = np.random.default_rng(seed)
    k_n = rand_unit_complex(rng, (d,))
    v_n = rand_unit_complex(rng, (d,))
    baseline_S = outer_v_kstar(v_n, k_n)
    rows = []
    print('\n[A8.2] NIAH GSP protection sweep')
    for p_fill in protect_values:
        g_fill = gsp_gamma(dt, p_fill)
        for T in distances:
            S = baseline_S.copy()
            S = simulate_filler_steps(
                S, T, rng, d, g_fill, p_fill=p_fill, dt=dt,
                report_every=report_every,
            )
            rel = relative_retrieval(S, k_n, v_n)
            theory = (g_fill ** T) if p_fill >= 1.0 - 1e-12 else None
            rows.append({
                'p_fill': p_fill, 'gamma_fill': g_fill, 'distance': T,
                'relative': rel, 'theory_decay_only': theory,
            })
        sub = [r for r in rows if r['p_fill'] == p_fill and r['distance'] == max(distances)]
        if sub:
            print(f'  p_fill={p_fill:.2f} gamma={g_fill:.6f} T={max(distances):,} '
                  f'rel={sub[0]["relative"]:.4f}')
    return {'dt': dt, 'results': rows}


def niah_position_grid(
    d: int,
    lengths: Sequence[int],
    rel_positions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    gamma: float = 0.995,
    gsp_p: float = 0.0,
    dt: float | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    if dt is None:
        dt = math.log1p(math.exp(-4.0))
    g_fill = gsp_gamma(dt, gsp_p) if gsp_p > 0 else gamma
    rng = np.random.default_rng(seed)
    grid: List[List[float]] = []
    print(f'\n[A8.3] NIAH position×length grid (gamma={gamma}, gsp_p={gsp_p})')
    for T in lengths:
        row: List[float] = []
        for r in rel_positions:
            pos = int(r * max(T - 1, 0))
            k_n = rand_unit_complex(rng, (d,))
            v_n = rand_unit_complex(rng, (d,))
            S = np.zeros((d, d), dtype=np.complex128)
            for t in range(T + 1):
                if t == pos:
                    S = pam_step_additive(S, 1.0, v_n, k_n)
                elif t <= T:
                    k_f = rand_unit_complex(rng, (d,))
                    v_f = rand_unit_complex(rng, (d,))
                    v_eff = v_f * (1.0 - gsp_p)
                    S = pam_step_additive(S, g_fill, v_eff, k_f)
            rel = retrieve_score(S, k_n, v_n) / (retrieve_score(
                outer_v_kstar(v_n, k_n), k_n, v_n) + 1e-12)
            row.append(rel)
        grid.append(row)
        print(f'  T={T:4d}  ' + '  '.join(f'r={rp:.2f}:{v:.3f}' for rp, v in zip(rel_positions, row)))
    return {
        'lengths': list(lengths),
        'rel_positions': list(rel_positions),
        'grid': grid,
        'gamma': gamma,
        'gsp_p': gsp_p,
    }


def niah_multi_needle(
    d: int,
    n_needles: int,
    context_len: int,
    seed: int = 42,
    gamma: float = 0.995,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    keys, values = [], []
    positions = np.linspace(0, context_len, n_needles, dtype=int)
    S = np.zeros((d, d), dtype=np.complex128)
    needle_map = []
    for idx, pos in enumerate(positions):
        k_n = rand_unit_complex(rng, (d,))
        v_n = rand_unit_complex(rng, (d,))
        keys.append(k_n)
        values.append(v_n)
        needle_map.append({'index': idx, 'position': int(pos)})

    # Simulate: interleave filler and needles by position order
    events = [(int(positions[i]), 'needle', i) for i in range(n_needles)]
    events.sort()
    t = 0
    ev_ptr = 0
    while t <= context_len:
        if ev_ptr < len(events) and events[ev_ptr][0] == t:
            _, _, ni = events[ev_ptr]
            S = pam_step_additive(S, gamma, values[ni], keys[ni])
            ev_ptr += 1
        else:
            k_f = rand_unit_complex(rng, (d,))
            v_f = rand_unit_complex(rng, (d,))
            S = pam_step_additive(S, gamma, v_f, k_f)
        t += 1

    scores = []
    for i in range(n_needles):
        s = retrieve_score(S, keys[i], values[i])
        s0 = retrieve_score(outer_v_kstar(values[i], keys[i]), keys[i], values[i])
        scores.append({'needle': i, 'position': int(positions[i]), 'relative': s / (s0 + 1e-12)})

    print(f'\n[A8.4] Multi-needle k={n_needles} context={context_len}')
    for sc in scores:
        print(f'  needle@{sc["position"]:4d} rel={sc["relative"]:.4f}')
    return {'n_needles': n_needles, 'context_len': context_len, 'scores': scores}


def niah_key_collision(
    d: int,
    distance: int,
    cos_targets: Sequence[float] = (0.0, 0.5, 0.9),
    gamma: float = 0.995,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = []
    print(f'\n[A8.5] NIAH key-collision (T={distance})')
    k_n = rand_unit_complex(rng, (d,))
    v_n = rand_unit_complex(rng, (d,))
    for cos_t in cos_targets:
        S = outer_v_kstar(v_n, k_n)
        for _ in range(distance):
            k_f = rand_unit_complex(rng, (d,))
            if cos_t > 0:
                # Blend toward needle key
                k_f = k_f / (np.linalg.norm(k_f) + 1e-12)
                k_n_u = k_n / (np.linalg.norm(k_n) + 1e-12)
                k_blend = (1 - cos_t) * k_f + cos_t * k_n_u
                k_f = k_blend / (np.linalg.norm(k_blend) + 1e-12)
            v_f = rand_unit_complex(rng, (d,))
            S = pam_step_additive(S, gamma, v_f, k_f)
        rel = retrieve_score(S, k_n, v_n) / (retrieve_score(
            outer_v_kstar(v_n, k_n), k_n, v_n) + 1e-12)
        rows.append({'cos_to_needle': cos_t, 'relative': rel})
        print(f'  cos={cos_t:.1f} rel={rel:.4f}')
    return {'distance': distance, 'results': rows}


def test_niah(
    d: int = 64,
    distances: Sequence[int] = (64, 128, 256, 512, 1024, 2048),
    protect_values: Sequence[float] = (0.0, 0.5, 0.9, 0.99, 1.0),
    seed: int = 42,
) -> Dict[str, Any]:
    g_default = default_gamma_from_bias(-4.0)
    gammas = (g_default, 0.99, 0.995, 0.999, 1.0)
    return {
        'bare_decay': niah_bare_decay(d, distances, gammas, seed),
        'gsp_sweep': niah_gsp_sweep(d, distances, protect_values, seed=seed),
        'multi_needle_4': niah_multi_needle(d, 4, max(distances), seed=seed),
        'key_collision': niah_key_collision(d, 256, seed=seed),
    }


def test_niah_grid(
    d: int = 64,
    lengths: Sequence[int] = (128, 256, 512, 1024, 2048),
    modes: Sequence[str] = ('baseline', 'gsp', 'e3'),
    seed: int = 42,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    print('\n[A8.3/6] NIAH position grid — mode comparison')
    if 'baseline' in modes:
        out['baseline'] = niah_position_grid(d, lengths, gamma=0.995, gsp_p=0.0, seed=seed)
    if 'gsp' in modes:
        out['gsp'] = niah_position_grid(d, lengths, gamma=0.995, gsp_p=0.99, seed=seed)
    if 'e3' in modes:
        # Slow state approximated by high gamma on grid (E3 math uses multi-state in interference)
        out['e3_slow_gamma'] = niah_position_grid(d, lengths, gamma=0.999, gsp_p=0.0, seed=seed)
    return out


# ── A9: Long-context math (beyond training seq_len=2048) ───────────────────────

def test_long_context(
    d: int = 64,
    max_distance: int = 262144,
    seed: int = 42,
    simulate_filler: bool = True,
) -> Dict[str, Any]:
    """PAM memory math at 4K–1M+ tokens — no LM, no checkpoint.

    Unlike V11 training (seq_len=2048), the recurrent PAM state has no fixed
    context cap at inference: state is O(d²) per layer regardless of T.
    This test measures decay/interference ceilings at extreme distances.
    """
    distances = long_context_distance_sweep(max_distance)
    dt = default_dt()
    g_default = default_gamma_from_bias(-4.0)
    gammas = (g_default, 0.99, 0.995, 0.999, 1.0)
    protect_values = (0.0, 0.9, 0.99, 1.0)
    report_every = 50000 if max(distances) > 50000 else 0

    print('\n[A9] Long-context PAM math')
    print(f'  d={d}, distances={distances[0]:,}..{distances[-1]:,} ({len(distances)} points)')
    print(f'  Note: PAM recurrent state is O(d²) — no KV cache growth; T is unbounded in math.')

    out: Dict[str, Any] = {
        'd': d,
        'max_distance': max_distance,
        'distances': list(distances),
        'simulate_filler': simulate_filler,
    }

    # 1) Pure decay envelope (analytic) — no filler, exact γ^T
    analytic_rows = []
    print('\n  [A9.1] Analytic decay-only envelope (no filler writes, exact γ^T)')
    for gamma in gammas:
        for T in distances:
            analytic_rows.append({
                'gamma': gamma,
                'distance': T,
                'relative': gamma ** T,
                'mode': 'analytic_decay_only',
            })
        r2048 = gamma ** 2048
        r_max = gamma ** distances[-1]
        print(f'    gamma={gamma:.4f}: γ^2048={r2048:.2e}, γ^{distances[-1]:,}={r_max:.2e}')
    out['analytic_decay'] = analytic_rows

    # 2) Simulated NIAH with random filler (interference + decay)
    if simulate_filler:
        out['niah_filler'] = niah_bare_decay(
            d, distances, gammas, seed=seed, report_every=report_every,
        )
        out['gsp'] = niah_gsp_sweep(
            d, distances, protect_values, dt=dt, seed=seed + 1,
            report_every=report_every,
        )

    # 3) Early vs late needle at the longest context (recency bias check)
    rng = np.random.default_rng(seed + 2)
    long_T = distances[-1]
    rel_positions = (0.0, 1.0)  # first vs last token only — full grid is too slow at 256K+
    print(f'\n  [A9.3] Early vs late needle at T={long_T:,}')
    needle_rows = []
    for gsp_p in (0.0, 0.99):
        g_fill = gsp_gamma(dt, gsp_p) if gsp_p > 0 else 0.995
        for r in rel_positions:
            pos = int(r * max(long_T - 1, 0))
            k_n = rand_unit_complex(rng, (d,))
            v_n = rand_unit_complex(rng, (d,))
            S = np.zeros((d, d), dtype=np.complex128)
            for t in range(long_T + 1):
                if t == pos:
                    S = pam_step_additive(S, 1.0, v_n, k_n)
                elif t <= long_T:
                    k_f = rand_unit_complex(rng, (d,))
                    v_f = rand_unit_complex(rng, (d,))
                    v_eff = v_f * (1.0 - gsp_p)
                    S = pam_step_additive(S, g_fill, v_eff, k_f)
                if report_every and t > 0 and t % report_every == 0:
                    print(f'    ... position sim t={t:,}/{long_T:,}', flush=True)
            rel = relative_retrieval(S, k_n, v_n)
            needle_rows.append({
                'distance': long_T, 'rel_position': r, 'position': pos,
                'gsp_p': gsp_p, 'relative': rel,
            })
            print(f'    gsp_p={gsp_p:.2f} r={r:.2f} (pos={pos:,}) rel={rel:.4f}')
    out['needle_early_late'] = needle_rows

    # 4) Summary: max distance with ≥90% retrieval under each mode
    summary = []
    for label, key, filt in (
        ('gsp_p=1.0 decay-only', 'gsp', lambda r: r.get('p_fill') == 1.0),
        ('gamma=0.999 analytic', 'analytic_decay', lambda r: r.get('gamma') == 0.999),
    ):
        rows = out.get(key, {})
        if isinstance(rows, dict):
            rows = rows.get('results', rows)
        eligible = [r for r in rows if filt(r) and r.get('relative', 0) >= 0.9]
        max_d = max((r['distance'] for r in eligible), default=0)
        summary.append({'mode': label, 'max_distance_ge_90pct': max_d})
    out['summary'] = summary
    print('\n  [A9 summary] max distance at ≥90% relative retrieval:')
    for s in summary:
        print(f'    {s["mode"]}: T={s["max_distance_ge_90pct"]:,}')

    return out


# ── Runner / CLI ─────────────────────────────────────────────────────────────

ALL_TESTS = (
    'binding', 'persistence', 'interference', 'rank',
    'conjugate', 'layer-bridge', 'niah', 'niah-grid', 'long-context',
)


def run_all(output_dir: Path, seed: int = 42) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'seed': seed,
        'tests': {},
    }
    results['tests']['binding'] = test_binding(seed=seed)
    results['tests']['persistence'] = test_persistence(seed=seed)
    results['tests']['interference_additive'] = test_interference(seed=seed, use_delta=False)
    results['tests']['interference_delta'] = test_interference(seed=seed, use_delta=True)
    results['tests']['interference_e3'] = test_interference(seed=seed, n_states=3)
    results['tests']['rank_random'] = test_rank(seed=seed, mode='random')
    results['tests']['rank_overwrite'] = test_rank(seed=seed, mode='overwrite')
    results['tests']['conjugate'] = test_conjugate_interference(seed=seed)
    results['tests']['layer_bridge'] = test_layer_bridge(seed=seed)
    results['tests']['niah'] = test_niah(seed=seed)
    results['tests']['niah_grid'] = test_niah_grid(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'pam_math_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with out_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')
    return results


def _parse_floats(s: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(',') if x.strip())


def _parse_ints(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(',') if x.strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description='V11 PAM math probes (no checkpoint)')
    p.add_argument('--all', action='store_true', help='Run full battery')
    p.add_argument('--test', type=str, default='', help=f'One of: {",".join(ALL_TESTS)}')
    p.add_argument('--output-dir', type=str, default='logs/v11/pam_math')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--distances', type=str, default='64,128,256,512,1024,2048')
    p.add_argument('--gamma', type=str, default='0.99,0.995,0.999,1.0')
    p.add_argument('--protect', type=str, default='0,0.5,0.9,0.99,1.0')
    p.add_argument('--lengths', type=str, default='128,256,512,1024,2048')
    p.add_argument('--modes', type=str, default='baseline,e1,e2,e3')
    p.add_argument('--max-n', type=int, default=200)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--pairs', type=int, default=32)
    p.add_argument('--filler', type=int, default=256)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--max-distance', type=int, default=262144,
                   help='Long-context sweep ceiling (default 256K; try 1048576 for 1M)')
    p.add_argument('--no-filler-sim', action='store_true',
                   help='Long-context: analytic decay only, skip filler simulation')
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)

    if args.all:
        run_all(out_dir, seed=args.seed)
        return 0

    if not args.test:
        p.print_help()
        return 1

    test = args.test.replace('_', '-')
    result: Dict[str, Any] = {}

    if test == 'binding':
        result = test_binding(max_n=args.max_n, trials=args.trials, seed=args.seed)
    elif test == 'persistence':
        result = test_persistence(
            distances=_parse_ints(args.distances),
            gammas=_parse_floats(args.gamma),
            seed=args.seed,
        )
    elif test == 'interference':
        result = test_interference(
            pair_counts=(4, 8, 16, args.pairs),
            filler_counts=(0, 64, args.filler, 1024),
            seed=args.seed,
        )
    elif test == 'rank':
        result = test_rank(steps=args.steps, seed=args.seed)
    elif test == 'conjugate':
        result = test_conjugate_interference(seed=args.seed)
    elif test == 'layer-bridge':
        result = test_layer_bridge(modes=tuple(args.modes.split(',')), seed=args.seed)
    elif test == 'niah':
        result = test_niah(
            distances=_parse_ints(args.distances),
            protect_values=_parse_floats(args.protect),
            seed=args.seed,
        )
    elif test == 'niah-grid':
        result = test_niah_grid(
            lengths=_parse_ints(args.lengths),
            modes=tuple(args.modes.split(',')),
            seed=args.seed,
        )
    elif test == 'long-context':
        result = test_long_context(
            max_distance=args.max_distance,
            seed=args.seed,
            simulate_filler=not args.no_filler_sim,
        )
    else:
        print(f'Unknown test: {test}', file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'pam_math_{test}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with out_path.open('w') as f:
        json.dump({'test': test, 'result': result, 'seed': args.seed}, f, indent=2)
    print(f'\nResults saved to {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
