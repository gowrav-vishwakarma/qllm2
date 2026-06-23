"""Multi-association interference, conjugate retrieval, and layer bridge."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from memory_probes.core import (
    abs_inner,
    default_gamma_from_bias,
    mat_vec,
    outer_v_kstar,
    pam_step_additive,
    pam_step_delta,
    rand_unit_complex,
    retrieve_score,
)
from v11.model import V11Config, V11PAMLayer


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
    print(f'\n[interference] Multi-association (delta={use_delta}, n_states={n_states})')

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


def test_conjugate_interference(
    d: int = 64,
    T: int = 128,
    seed: int = 42,
    layer_bridge: bool = True,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    Q = rand_unit_complex(rng, (T, d))
    K = rand_unit_complex(rng, (T, d))

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
    print('\n[interference] Conjugate retrieval (K*·Q)')
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


def _numpy_recur_additive(S_np, g, v_np, k_np, q_np):
    S = S_np.copy()
    S = pam_step_additive(S, g, v_np, k_np)
    y = mat_vec(S, q_np)
    return y, S


def _torch_from_c(z: np.ndarray, device='cpu') -> torch.Tensor:
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
    print('\n[interference] V11PAMLayer synthetic bridge (NumPy vs PyTorch recurrent)')
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
