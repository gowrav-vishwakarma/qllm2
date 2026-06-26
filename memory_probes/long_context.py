"""Needle-in-haystack and long-context sweeps for recurrent matrix memory."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np

from memory_probes.core import (
    default_dt,
    default_gamma_from_bias,
    gsp_gamma,
    long_context_distance_sweep,
    outer_v_kstar,
    pam_step_additive,
    rand_unit_complex,
    relative_retrieval,
    retrieve_score,
    simulate_filler_steps,
)


def niah_bare_decay(
    d: int,
    distances: Sequence[int],
    gammas: Sequence[float],
    seed: int = 42,
    report_every: int = 0,
    adapter=None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    k_n = rand_unit_complex(rng, (d,))
    v_n = rand_unit_complex(rng, (d,))
    baseline_S = outer_v_kstar(v_n, k_n)
    rows = []
    arch = getattr(adapter, 'name', 'pam') if adapter is not None else 'pam'
    print(f'\n[long-context] NIAH bare decay envelope ({arch})')
    for gamma in gammas:
        for T in distances:
            if adapter is not None:
                from memory_probes.adapters import adapter_relative_retrieval
                adapter.reset()
                adapter.write(k_n, v_n)
                for _ in range(T):
                    k_f = rand_unit_complex(rng, (d,))
                    v_f = rand_unit_complex(rng, (d,))
                    adapter.write(k_f, v_f, gamma)
                rel = adapter_relative_retrieval(adapter, k_n, v_n, d)
                rows.append({'gamma': gamma, 'distance': T, 'relative': rel,
                             'theory_gamma_T': gamma ** T, 'decay_only_relative': gamma ** T})
                continue
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
    print('\n[long-context] NIAH GSP protection sweep')
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
    print(f'\n[long-context] NIAH position×length grid (gamma={gamma}, gsp_p={gsp_p})')
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
    for idx, pos in enumerate(positions):
        k_n = rand_unit_complex(rng, (d,))
        v_n = rand_unit_complex(rng, (d,))
        keys.append(k_n)
        values.append(v_n)

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

    print(f'\n[long-context] Multi-needle k={n_needles} context={context_len}')
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
    print(f'\n[long-context] NIAH key-collision (T={distance})')
    k_n = rand_unit_complex(rng, (d,))
    v_n = rand_unit_complex(rng, (d,))
    for cos_t in cos_targets:
        S = outer_v_kstar(v_n, k_n)
        for _ in range(distance):
            k_f = rand_unit_complex(rng, (d,))
            if cos_t > 0:
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
    adapter=None,
) -> Dict[str, Any]:
    g_default = default_gamma_from_bias(-4.0)
    gammas = (g_default, 0.99, 0.995, 0.999, 1.0)
    if adapter is not None:
        # Adapter NIAH: bare-decay sweep only. GSP/multi-needle are PAM-specific
        # write-gating mechanisms not exposed by the generic adapter interface.
        return {
            'arch': getattr(adapter, 'name', 'adapter'),
            'bare_decay': niah_bare_decay(d, distances, gammas, seed, adapter=adapter),
        }
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
    print('\n[long-context] NIAH position grid — mode comparison')
    if 'baseline' in modes:
        out['baseline'] = niah_position_grid(d, lengths, gamma=0.995, gsp_p=0.0, seed=seed)
    if 'gsp' in modes:
        out['gsp'] = niah_position_grid(d, lengths, gamma=0.995, gsp_p=0.99, seed=seed)
    if 'e3' in modes:
        out['e3_slow_gamma'] = niah_position_grid(d, lengths, gamma=0.999, gsp_p=0.0, seed=seed)
    return out


def test_long_context(
    d: int = 64,
    max_distance: int = 262144,
    seed: int = 42,
    simulate_filler: bool = True,
) -> Dict[str, Any]:
    """Recurrent matrix memory at 4K–1M+ tokens — no LM, no checkpoint."""
    distances = long_context_distance_sweep(max_distance)
    dt = default_dt()
    g_default = default_gamma_from_bias(-4.0)
    gammas = (g_default, 0.99, 0.995, 0.999, 1.0)
    protect_values = (0.0, 0.9, 0.99, 1.0)
    report_every = 50000 if max(distances) > 50000 else 0

    print('\n[long-context] Extreme distance sweep')
    print(f'  d={d}, distances={distances[0]:,}..{distances[-1]:,} ({len(distances)} points)')
    print('  Note: recurrent state is O(d²) — no KV cache growth; T is unbounded in math.')

    out: Dict[str, Any] = {
        'd': d,
        'max_distance': max_distance,
        'distances': list(distances),
        'simulate_filler': simulate_filler,
    }

    analytic_rows = []
    print('\n  Analytic decay-only envelope (no filler writes, exact γ^T)')
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

    if simulate_filler:
        out['niah_filler'] = niah_bare_decay(
            d, distances, gammas, seed=seed, report_every=report_every,
        )
        out['gsp'] = niah_gsp_sweep(
            d, distances, protect_values, dt=dt, seed=seed + 1,
            report_every=report_every,
        )

    rng = np.random.default_rng(seed + 2)
    long_T = distances[-1]
    rel_positions = (0.0, 1.0)
    print(f'\n  Early vs late needle at T={long_T:,}')
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
    print('\n  Summary: max distance at ≥90% relative retrieval:')
    for s in summary:
        print(f'    {s["mode"]}: T={s["max_distance_ge_90pct"]:,}')

    return out
