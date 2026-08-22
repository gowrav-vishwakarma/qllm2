"""Verify the numpy hypothesis-lab mirror against the REAL V13 model code.

Three checks:

A) step-equivalence: for a random input stream, the numpy reference dynamics
   (the convention used in mem_dynamics.py: S <- g*S; pred = S@k;
    u = beta*(v_prot - pred); S += u k*) must match
   V13PAMLayer._recur_step_delta / _recur_step_additive exactly.
   Proves the synthetic experiments model the actual layer math.

B) train/decode consistency: with delta + n_states=3 (the V13 production
   path), the chunked parallel forward (_forward_multistate -> _forward_delta
   UT transform) and the O(1) recurrent forward (_recurrent ->
   _recur_step_delta) must produce the same outputs and final states.
   A mismatch means training and inference run different dynamics.

C) leak measurement on the real layer: run the real recurrent step on a
   synthetic protected-fact stream (fact written, then protected filler)
   and measure the fact-signal decay, comparing write_mode='delta' vs
   write_mode='additive'. This measures the erosion on the ACTUAL model
   (real projections, real GSP gate, real dt sampling).

Run:  python -m v13.experiments.layer_equivalence
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from v13.complex_ops import cabs
from v13.model import V13Config, V13PAMLayer


def make_layer(write_mode: str = 'delta', n_states: int = 3, seed: int = 0,
               erase_gate: bool = False) -> V13PAMLayer:
    torch.manual_seed(seed)
    cfg = V13Config(
        vocab_size=256, dim=96, n_heads=3, head_dim=32, n_layers=1,
        expand=3, dropout=0.0, max_seq_len=512,
        use_learned_pos=False, use_rope=True, use_gsp=True, fused_qkv=True,
        qk_norm=False, tie_weights=True, gradient_checkpointing=False,
        activation='swish', chunk_size=64, decay_mode='head',
        write_mode=write_mode, n_states=n_states, delta_chunk=32,
        delta_erase_gate=erase_gate,
        state_dt_spread=2.0, base_dt_bias=-4.0,
        gate_content_aware=True, protect_gate_bias=-3.0,
        routing_content_aware=False, state_compete=False,
        phase_init='zero', route_balance_lambda=0.0, aux_loss_weight=1.0,
        fused_e3=False, recompute_pam_chunks=False, gamma_floor=0.0,
        gate_surprisal_lambda=0.0,
        vault_state=True, vault_state_idx=0, write_phase_address=True,
    )
    layer = V13PAMLayer(cfg, layer_idx=0)
    layer.eval()
    return layer


def to_complex_np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().float()
    return t[..., 0].numpy() + 1j * t[..., 1].numpy()


# ─────────────────────────────────────────────────────────────────────────────
# A) numpy reference vs _recur_step_*
# ─────────────────────────────────────────────────────────────────────────────

def check_step_equivalence(write_mode: str, seq_len: int = 48, seed: int = 0) -> Dict:
    layer = make_layer(write_mode=write_mode, n_states=1, seed=seed)
    torch.manual_seed(100 + seed)
    B, T, dim = 2, seq_len, layer.dim
    x = torch.randn(B, T, dim, 2)
    with torch.no_grad():
        queries, keys, values = layer._project(x, 0)
        decay_gamma, protected_values = layer._gamma_and_vprime(x, values)
        write_beta = torch.sigmoid(layer.beta_proj(cabs(x))).transpose(1, 2) if write_mode == 'delta' else None

        query_scale = layer.head_dim ** -0.5
        S = torch.zeros(B, layer.num_heads, layer.head_dim, layer.head_dim, 2)
        for t in range(T):
            if write_mode == 'delta':
                _, S = layer._recur_step_delta(
                    S, decay_gamma[:, :, t], protected_values[:, :, t],
                    keys[:, :, t], queries[:, :, t] * query_scale, write_beta[:, :, t],
                )
            else:
                _, S = layer._recur_step_additive(
                    S, decay_gamma[:, :, t], protected_values[:, :, t],
                    keys[:, :, t], queries[:, :, t] * query_scale,
                )

        S_np = np.zeros((B, layer.num_heads, layer.head_dim, layer.head_dim), dtype=complex)
        g = decay_gamma.detach().cpu().numpy()
        vprot = to_complex_np(protected_values)
        k_np = to_complex_np(keys)
        beta = write_beta.detach().cpu().numpy() if write_beta is not None else 1.0
        for t in range(T):
            S_np = g[:, :, t][:, :, None, None] * S_np
            pred = np.einsum('bhij,bhj->bhi', S_np, k_np[:, :, t])
            if write_mode == 'delta':
                u = beta[:, :, t][:, :, None] * (vprot[:, :, t] - pred)
            else:
                u = vprot[:, :, t]
            S_np = S_np + np.einsum('bhi,bhj->bhij', u, np.conj(k_np[:, :, t]))

    max_err = float(np.max(np.abs(to_complex_np(S) - S_np)))
    scale = float(np.max(np.abs(to_complex_np(S))))
    return {'write_mode': write_mode, 'max_abs_err': max_err, 'state_scale': scale,
            'rel_err': max_err / max(scale, 1e-9)}


# ─────────────────────────────────────────────────────────────────────────────
# B) chunked (UT) forward vs recurrent forward, multistate delta
# ─────────────────────────────────────────────────────────────────────────────

def check_chunked_vs_recurrent(seq_len: int = 96, seed: int = 0) -> Dict:
    layer = make_layer(write_mode='delta', n_states=3, seed=seed)
    torch.manual_seed(200 + seed)
    B, T, dim = 1, seq_len, layer.dim
    x = torch.randn(B, T, dim, 2)
    K, H, d = layer.n_states, layer.num_heads, layer.head_dim
    with torch.no_grad():
        out_chunked, state_chunked = layer(x)                       # UT / K-loop path
        zero_state = torch.zeros(K, B, H, d, d, 2)
        out_recur, state_recur = layer(x, state=zero_state)         # O(1) recurrent path

    o_err = float(torch.max(torch.abs(out_chunked - out_recur)))
    s_err = float(torch.max(torch.abs(state_chunked - state_recur)))
    o_scale = float(torch.max(torch.abs(out_chunked)))
    s_scale = float(torch.max(torch.abs(state_chunked)))
    return {
        'seq_len': T, 'out_max_err': o_err, 'out_rel_err': o_err / max(o_scale, 1e-9),
        'state_max_err': s_err, 'state_rel_err': s_err / max(s_scale, 1e-9),
        'state_scale': s_scale,
    }


# ─────────────────────────────────────────────────────────────────────────────
# C) real-layer leak: protected fact + protected filler through _recur_step_*
# ─────────────────────────────────────────────────────────────────────────────

def check_real_layer_leak(write_mode: str = 'delta', steps: int = 512, seed: int = 0,
                          erase_gate: bool = False, erase_beta_val: float = None) -> Dict:
    """Write one fact (p=0), then run `steps` of protected filler through the
    real layer's recurrent step. Track the fact-signal magnitude per head."""
    layer = make_layer(write_mode=write_mode, n_states=1, seed=seed, erase_gate=erase_gate)
    if erase_beta_val is None:
        erase_beta_val = 0.9
    torch.manual_seed(300 + seed)
    B, dim = 1, layer.dim
    H, d = layer.num_heads, layer.head_dim
    query_scale = d ** -0.5

    with torch.no_grad():
        # fact token (no gate: use raw values, gamma=1 handled by feeding a token whose
        # protect prob we force to 0 via a direct step)
        x_fact = torch.randn(B, 1, dim, 2)
        qf, kf, vf = layer._project(x_fact, 0)
        # force p=0 for the fact: decay=base, v_prot = v
        decay_fact = torch.full((B, H), 1.0)
        S = torch.zeros(B, H, d, d, 2)
        if write_mode == 'delta':
            beta_t = torch.full((B, H), 0.9)
            _, S = layer._recur_step_delta(S, decay_fact, vf[:, :, 0], kf[:, :, 0],
                                           qf[:, :, 0] * query_scale, beta_t,
                                           erase_beta_t=torch.full((B, H), erase_beta_val))
        else:
            _, S = layer._recur_step_additive(S, decay_fact, vf[:, :, 0], kf[:, :, 0],
                                              qf[:, :, 0] * query_scale)
        fact_key = kf[:, :, 0]
        fact_val = vf[:, :, 0]
        # measure signal: |Re( (S k0)^H v0 )| per head
        def signal(Sm):
            real, imag = Sm[..., 0], Sm[..., 1]
            kr, ki = fact_key[..., 0], fact_key[..., 1]
            vr, vi = fact_val[..., 0], fact_val[..., 1]
            # (S k0) = (Sr + i Si)(kr + i ki)
            sk_r = torch.einsum('bhij,bhj->bhi', real, kr) - torch.einsum('bhij,bhj->bhi', imag, ki)
            sk_i = torch.einsum('bhij,bhj->bhi', real, ki) + torch.einsum('bhij,bhj->bhi', imag, kr)
            # <sk, v> summed over d (real part of vdot), per head
            s = torch.einsum('bhd,bhd->bh', sk_r, vr) + torch.einsum('bhd,bhd->bh', sk_i, vi)
            return s.abs().cpu().numpy()

        tr = [signal(S)]
        for t in range(steps):
            x_t = torch.randn(B, 1, dim, 2)
            qt, kt, vt = layer._project(x_t, t + 1)
            # force PROTECTED filler: p=1 -> decay=1, v_prot=0 (pure read + erase only)
            decay_t = torch.ones(B, H)
            v_prot_t = torch.zeros_like(vt[:, :, 0])
            if write_mode == 'delta':
                beta_t = torch.full((B, H), 0.9)
                _, S = layer._recur_step_delta(S, decay_t, v_prot_t, kt[:, :, 0],
                                               qt[:, :, 0] * query_scale, beta_t,
                                               erase_beta_t=torch.full((B, H), erase_beta_val))
            else:
                _, S = layer._recur_step_additive(S, decay_t, v_prot_t, kt[:, :, 0],
                                                  qt[:, :, 0] * query_scale)
            tr.append(signal(S))
    tr = np.array(tr)
    checks = (64, 128, 256, 512)
    return {
        'write_mode': write_mode,
        'erase_beta': erase_beta_val,
        'signal': {str(c): float(tr[c].mean()) for c in checks if c < len(tr)},
        'final': float(tr[-1].mean()),
        'log_decay_per_step': float(np.log(max(tr[-1].mean(), 1e-9) / max(tr[0].mean(), 1e-9)) / (len(tr) - 1)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# D) E2b backward compatibility: two-gate layer with erase==write == legacy
# E2b) chunked vs recurrent consistency WITH the erase gate on
# ─────────────────────────────────────────────────────────────────────────────

def check_erase_gate_backward_compat(seed: int = 0) -> Dict:
    """A two-gate layer whose erase_beta_proj is set equal to beta_proj must
    reproduce the legacy layer bit-for-bit (same input, same seeds)."""
    legacy = make_layer('delta', n_states=3, seed=seed, erase_gate=False)
    gated = make_layer('delta', n_states=3, seed=seed, erase_gate=True)
    with torch.no_grad():
        gated.erase_beta_proj.weight.copy_(legacy.beta_proj.weight)
        gated.erase_beta_proj.bias.copy_(legacy.beta_proj.bias)
    torch.manual_seed(900 + seed)
    x = torch.randn(1, 64, legacy.dim, 2)
    with torch.no_grad():
        o_leg, s_leg = legacy(x)
        o_gat, s_gat = gated(x)
    return {
        'out_max_diff': float(torch.max(torch.abs(o_leg - o_gat))),
        'state_max_diff': float(torch.max(torch.abs(s_leg - s_gat))),
    }


def check_erase_gate_chunked_vs_recurrent(seq_len: int = 96, seed: int = 0) -> Dict:
    layer = make_layer('delta', n_states=3, seed=seed, erase_gate=True)
    torch.manual_seed(950 + seed)
    B, T, dim = 1, seq_len, layer.dim
    x = torch.randn(B, T, dim, 2)
    K, H, d = layer.n_states, layer.num_heads, layer.head_dim
    with torch.no_grad():
        out_chunked, state_chunked = layer(x)
        zero_state = torch.zeros(K, B, H, d, d, 2)
        out_recur, state_recur = layer(x, state=zero_state)
    o_scale = float(torch.max(torch.abs(out_chunked)))
    s_scale = float(torch.max(torch.abs(state_chunked)))
    return {
        'out_rel_err': float(torch.max(torch.abs(out_chunked - out_recur))) / max(o_scale, 1e-9),
        'state_rel_err': float(torch.max(torch.abs(state_chunked - state_recur))) / max(s_scale, 1e-9),
    }


def main():
    print('== A) numpy reference vs real _recur_step_* ==')
    for wm in ('additive', 'delta'):
        r = check_step_equivalence(wm)
        ok = r['rel_err'] < 1e-4
        print(f"  {wm:<9} max_abs_err={r['max_abs_err']:.2e}  rel_err={r['rel_err']:.2e}  {'OK' if ok else 'MISMATCH!'}")

    print('== B) chunked (UT) vs recurrent, multistate delta ==')
    r = check_chunked_vs_recurrent(seq_len=96)
    ok = r['out_rel_err'] < 1e-3 and r['state_rel_err'] < 1e-3
    print(f"  out_rel_err={r['out_rel_err']:.2e}  state_rel_err={r['state_rel_err']:.2e}  "
          f"(state_scale={r['state_scale']:.1f})  {'OK' if ok else 'MISMATCH!'}")

    print('== C) real-layer leak: protected fact + protected filler (p=1, write beta=0.9) ==')
    for wm, eg, eb in [('additive', False, None), ('delta', False, 0.9),
                       ('delta', True, 0.047), ('delta', True, 0.0)]:
        r = check_real_layer_leak(wm, steps=512, erase_gate=eg, erase_beta_val=eb)
        s = '  '.join(f't{c}={v:.3f}' for c, v in r['signal'].items())
        tag = wm + ('+erase_gate' if eg else '') + (f'(e={eb})' if eg else '')
        print(f"  {tag:<20} {s}  logdecay/step={r['log_decay_per_step']:+.5f}")
    print('  (additive retains flat: erase-free. legacy delta erodes ~beta/d per step.')
    print('   E2b: low-init erase gate (0.047) must cut the erosion by ~19x)')

    print('== D) E2b backward compat: erase==write gates must equal legacy ==')
    r = check_erase_gate_backward_compat()
    ok = r['out_max_diff'] < 1e-6 and r['state_max_diff'] < 1e-6
    print(f"  out_diff={r['out_max_diff']:.2e}  state_diff={r['state_max_diff']:.2e}  {'OK' if ok else 'MISMATCH!'}")

    print('== E) chunked vs recurrent WITH erase gate ==')
    r = check_erase_gate_chunked_vs_recurrent()
    ok = r['out_rel_err'] < 1e-3 and r['state_rel_err'] < 1e-3
    print(f"  out_rel_err={r['out_rel_err']:.2e}  state_rel_err={r['state_rel_err']:.2e}  {'OK' if ok else 'MISMATCH!'}")


if __name__ == '__main__':
    main()
