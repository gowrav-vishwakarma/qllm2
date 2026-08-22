"""V13 memory-dynamics hypothesis lab.

Re-runnable experiments for the fact-recall problem. Everything here mirrors
the V13 model conventions exactly (see layer_equivalence.py, which verifies
the mirror against the real V13PAMLayer code):

  * per token t:  S <- gamma_t * S          (gamma = base*(1-p) + p, p = GSP protect prob)
  * pred = S @ k  (readout AFTER decay — the model convention)
  * v_prot = (1-p) * v                      (GSP write suppression, applied upstream)
  * beta = write gate in (0,1)              (sigmoid projection of |x| in the model)

Variants (write dynamics), all S in C^{d x d}:
  additive         S += v_prot k*
  delta_v13        S += beta (v_prot - pred) k*          <- current V13 default
  delta_erase_ws   S += beta (v_prot - ws*pred) k*
  delta_erase_ws2  S += beta (v_prot - ws^2*pred) k*
  delta_twogate    S += (beta_w*v_prot - beta_e*pred) k*  <- proposed fix: learned erase gate
  delta_meanfix    gamma' = gamma/(1-beta/d); S = gamma'*S + beta (v_prot-pred) k*
                   (mean-subtracted delta; expected to explode via Lyapunov term)

Modifiers (the "sigmoid competition" hypothesis):
  sat_tanh         S <- tau*tanh(S/tau) every `chunk` steps (elementwise, real+imag)
  normcap          S <- S * min(1, C/||S||_F) every `chunk` steps

Regimes:
  random           unit complex k, v
  dc               k, v share a common DC direction (mimics real-text key bias)

Protocols:
  retention        1 fact + T protected filler -> signal |<S k0, v0>| decay curve
  recall           N facts + filler -> contrastive accuracy (capacity & gate sweeps)
  overwrite        same key, 10 values -> does the latest value win?
  rank             state effective rank + spectrum

Run:
  python -m v13.experiments.mem_dynamics --all
  python -m v13.experiments.mem_dynamics --protocol retention --variants delta_v13 delta_twogate
  python -m v13.experiments.mem_dynamics --protocol recall --quick
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

RESULTS_DIR = Path(__file__).parent / 'results'


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics
# ─────────────────────────────────────────────────────────────────────────────

def unit_complex(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return z / np.linalg.norm(z)


@dataclass
class Env:
    """Per-stream environment: DC direction (regime) and RNG."""
    rng: np.random.Generator
    d: int = 32
    dc: float = 0.0
    dcvec: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.dc > 0.0:
            self.dcvec = unit_complex(self.rng, self.d)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        d = self.d
        k = unit_complex(self.rng, d)
        v = unit_complex(self.rng, d)
        if self.dc > 0.0 and self.dcvec is not None:
            k = (k + self.dc * self.dcvec) / np.linalg.norm(k + self.dc * self.dcvec)
            v = (v + self.dc * self.dcvec) / np.linalg.norm(v + self.dc * self.dcvec)
        return k, v


def step(
    S: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    gamma: float,
    ws: float,
    beta_w: float,
    beta_e: float,
    variant: str,
    d: int,
) -> np.ndarray:
    """One token of memory dynamics. Returns new S (does not mutate in place)."""
    if variant == 'delta_meanfix':
        gamma = gamma / (1.0 - beta_w / d)
    S = gamma * S
    pred = S @ k
    v_prot = ws * v
    if variant == 'additive':
        u = v_prot
    elif variant == 'delta_v13':
        u = beta_w * (v_prot - pred)
    elif variant == 'delta_meanfix':
        u = beta_w * (v_prot - pred)
    elif variant == 'delta_erase_ws':
        u = beta_w * (v_prot - ws * pred)
    elif variant == 'delta_erase_ws2':
        u = beta_w * (v_prot - ws * ws * pred)
    elif variant == 'delta_twogate':
        u = beta_w * v_prot - beta_e * pred
    else:
        raise ValueError(f'unknown variant {variant}')
    return S + np.outer(u, np.conj(k))


def sat_tanh(S: np.ndarray, tau: float) -> np.ndarray:
    return tau * np.tanh(S / tau)


def normcap(S: np.ndarray, c: float) -> np.ndarray:
    n = np.linalg.norm(S)
    return S * min(1.0, c / max(n, 1e-9))


def apply_modifier(S: np.ndarray, modifier: str, t: int, chunk: int, tau: float, cap: float) -> np.ndarray:
    if t % chunk != 0:
        return S
    if modifier == 'sat':
        return sat_tanh(S, tau)
    if modifier == 'normcap':
        return normcap(S, cap)
    return S


def effective_rank(S: np.ndarray) -> float:
    sv = np.linalg.svd(S, compute_uv=False)
    s = sv / (sv.sum() + 1e-12)
    s = s[s > 1e-12]
    return float(np.exp(-np.sum(s * np.log(s)))) if s.size else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Protocol 1: retention (the leak measurement)
# ─────────────────────────────────────────────────────────────────────────────

def protocol_retention(
    variant: str,
    d: int = 32,
    base_gamma: float = 0.995,
    p_fill: float = 0.9,
    beta: float = 0.9,
    steps: int = 1024,
    checks: Tuple[int, ...] = (128, 256, 512, 1024),
    modifier: str = '',
    chunk: int = 32,
    tau: float = 2.0,
    cap: float = 4.0,
    dc: float = 0.0,
    erase_fire_prob: float = 0.0,
    seeds: int = 8,
) -> Dict:
    """One fact written, then `steps` protected filler tokens.

    beta_e (erase gate) is the ideal learned gate: 0 on filler, firing with
    `erase_fire_prob` to model gate mistakes.
    """
    ws = 1.0 - p_fill
    gamma_fill = base_gamma * ws + p_fill
    traces: List[np.ndarray] = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        env = Env(rng, d=d, dc=dc)
        k0, v0 = env.sample()
        S = np.outer(v0, np.conj(k0))  # fact written with p=0
        tr = [abs(np.vdot(S @ k0, v0))]
        for t in range(steps):
            k, v = env.sample()
            beta_e = beta if rng.random() < erase_fire_prob else 0.0
            S = step(S, k, v, gamma_fill, ws, beta, beta_e, variant, d)
            S = apply_modifier(S, modifier, t + 1, chunk, tau, cap)
            tr.append(abs(np.vdot(S @ k0, v0)))
        traces.append(np.array(tr))
    tr = np.mean(traces, axis=0)
    out = {
        'variant': variant,
        'modifier': modifier,
        'p_fill': p_fill,
        'steps': steps,
        'theory_pure_decay': {str(c): float(gamma_fill ** c) for c in checks},
        'signal': {str(c): float(tr[c]) for c in checks if c < len(tr)},
        'final_signal': float(tr[-1]),
        'eff_log_decay_per_step': float(
            np.log(max(tr[-1], 1e-12) / tr[0]) / (len(tr) - 1)
        ),
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Protocol 2: recall (capacity + gate sweep)
# ─────────────────────────────────────────────────────────────────────────────

def protocol_recall(
    variant: str,
    n_facts: int,
    p_fill: float,
    filler: int,
    d: int = 32,
    base_gamma: float = 0.995,
    beta: float = 0.9,
    n_cand: int = 16,
    modifier: str = '',
    chunk: int = 32,
    tau: float = 2.0,
    cap: float = 4.0,
    dc: float = 0.0,
    erase_fire_prob: float = 0.0,
    seed: int = 0,
) -> Dict:
    rng = np.random.default_rng(seed)
    env = Env(rng, d=d, dc=dc)
    ws = 1.0 - p_fill
    gamma_fill = base_gamma * ws + p_fill

    facts = [env.sample() for _ in range(n_facts)]
    cand_vals = [v for _, v in facts] + [env.sample()[1] for _ in range(max(0, n_cand - n_facts))]

    S = np.zeros((d, d), dtype=complex)
    for fact_k, fact_v in facts:
        for _ in range(8):
            k, v = env.sample()
            S = step(S, k, v, gamma_fill, ws, beta, 0.0, variant, d)
            S = apply_modifier(S, modifier, 1, chunk, tau, cap)
        S = step(S, fact_k, fact_v, 1.0, 1.0, beta, beta, variant, d)  # fact token: p=0
        S = apply_modifier(S, modifier, 1, chunk, tau, cap)
    rank_after_write = effective_rank(S)
    for t in range(filler):
        k, v = env.sample()
        beta_e = beta if rng.random() < erase_fire_prob else 0.0
        S = step(S, k, v, gamma_fill, ws, beta, beta_e, variant, d)
        S = apply_modifier(S, modifier, t + 1, chunk, tau, cap)

    correct = 0
    for i, (k, _) in enumerate(facts):
        y = S @ k
        sims = np.array([abs(np.vdot(y, cv)) for cv in cand_vals])
        if int(np.argmax(sims)) == i:
            correct += 1
    sv = np.linalg.svd(S, compute_uv=False)
    return {
        'variant': variant,
        'n_facts': n_facts,
        'p_fill': p_fill,
        'filler': filler,
        'acc': correct / n_facts,
        'rank_after_write': rank_after_write,
        'rank_final': effective_rank(S),
        'sv_top3': [round(float(x), 3) for x in sv[:3]],
        'sv_4to7': [round(float(x), 3) for x in sv[3:7]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Protocol 3: overwrite (delta's intended benefit — repeated keys)
# ─────────────────────────────────────────────────────────────────────────────

def protocol_overwrite(
    variant: str,
    d: int = 32,
    base_gamma: float = 0.995,
    beta: float = 0.9,
    n_values: int = 10,
    filler_between: int = 5,
    seed: int = 0,
) -> Dict:
    rng = np.random.default_rng(seed)
    env = Env(rng, d=d)
    k0 = unit_complex(rng, d)
    vals = [unit_complex(rng, d) for _ in range(n_values)]
    S = np.zeros((d, d), dtype=complex)
    for v in vals:
        for _ in range(filler_between):
            k, vf = env.sample()
            S = step(S, k, vf, base_gamma, 1.0, beta, 0.0, variant, d)
        S = step(S, k0, v, base_gamma, 1.0, beta, beta, variant, d)  # repeat key: erase on
    y = S @ k0
    sims = np.array([abs(np.vdot(y, vv)) for vv in vals])
    argmax = int(np.argmax(sims))
    return {'variant': variant, 'latest_wins': argmax == n_values - 1, 'argmax': argmax}


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_VARIANTS = [
    'additive',
    'delta_v13',
    'delta_erase_ws',
    'delta_erase_ws2',
    'delta_twogate',
    'delta_meanfix',
]


def run_retention(variants, quick=False) -> List[Dict]:
    steps = 512 if quick else 1024
    checks = (128, 256, 512) if quick else (128, 256, 512, 1024)
    seeds = 4 if quick else 8
    rows = []
    print(f'\n== retention: 1 fact + {steps} protected filler (p_fill=0.9) ==')
    print(f"{'variant':<18}" + ''.join(f' {str(c):>8}' for c in checks) + f" {'logdecay/step':>14}")
    for v in variants:
        for modifier in ([''] if quick else ['', 'sat']):
            r = protocol_retention(v, steps=steps, checks=checks, modifier=modifier, seeds=seeds)
            rows.append(r)
            name = v + (f'+{modifier}' if modifier else '')
            print(f'{name:<18}' + ''.join(f" {r['signal'].get(str(c), float('nan')):>8.3f}" for c in checks)
                  + f" {r['eff_log_decay_per_step']:>14.5f}")
    print(f"(theory pure-decay retention: " +
          ', '.join(f'{c}:{rows[0]["theory_pure_decay"][str(c)]:.3f}' for c in checks) + ')')
    return rows


def run_recall(variants, quick=False) -> List[Dict]:
    n_facts_list = [4, 8, 16] if quick else [4, 8, 16, 24, 32]
    p_fill_list = [0.9] if quick else [0.0, 0.6, 0.9]
    filler = 256 if quick else 512
    seeds = 2 if quick else 4
    rows = []
    print(f'\n== recall: contrastive accuracy, filler={filler}, candidates=16 ==')
    print(f"{'variant':<18} {'p_fill':>6} {'facts':>5} {'acc':>6} {'rank_f':>7}  sv_top3")
    for v in variants:
        for p_fill in p_fill_list:
            for n_facts in n_facts_list:
                accs, last = [], None
                for s in range(seeds):
                    r = protocol_recall(v, n_facts, p_fill, filler, seed=s)
                    accs.append(r['acc'])
                    last = r
                rows.append({**last, 'acc': float(np.mean(accs)), 'seeds': seeds})
                print(f'{v:<18} {p_fill:>6} {n_facts:>5} {np.mean(accs):>6.2f} {last["rank_final"]:>7.1f}  {last["sv_top3"]}')
    return rows


def run_overwrite(variants, quick=False) -> List[Dict]:
    seeds = 4 if quick else 8
    rows = []
    print(f'\n== overwrite: same key, 10 values, latest must win ==')
    for v in variants:
        res = [protocol_overwrite(v, seed=s) for s in range(seeds)]
        wins = sum(r['latest_wins'] for r in res)
        rows.append({'variant': v, 'latest_wins': wins, 'seeds': seeds})
        print(f'{v:<18} {wins}/{seeds} latest-wins')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--protocol', choices=['retention', 'recall', 'overwrite', 'all'], default='all')
    ap.add_argument('--variants', nargs='+', default=ALL_VARIANTS)
    ap.add_argument('--quick', action='store_true', help='reduced sweeps for fast iteration')
    ap.add_argument('--seeds', type=int, default=None)
    ap.add_argument('--out', type=str, default=None, help='JSON output path (default: results/)')
    args = ap.parse_args()

    t0 = time.time()
    result: Dict = {
        'protocol': args.protocol,
        'variants': args.variants,
        'quick': args.quick,
        'elapsed_s': None,
    }
    if args.protocol in ('retention', 'all'):
        result['retention'] = run_retention(args.variants, args.quick)
    if args.protocol in ('recall', 'all'):
        result['recall'] = run_recall(args.variants, args.quick)
    if args.protocol in ('overwrite', 'all'):
        result['overwrite'] = run_overwrite(args.variants, args.quick)
    result['elapsed_s'] = round(time.time() - t0, 1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f'mem_dynamics_{int(time.time())}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f'\nWrote {out_path} ({result["elapsed_s"]}s)')


if __name__ == '__main__':
    main()
