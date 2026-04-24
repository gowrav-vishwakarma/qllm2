"""First-principles bound on the gap between irreducible-loss floors of
complex-valued and real-valued language-modeling architectures, under the
assumptions of the quantum semantic framework (QSF, Agostino et al. 2025;
Agostino et al. 2026).

Framework
---------
The Bayes-optimal cross-entropy of any architecture decomposes as

    E = H(p) + D_KL(p || q*)

where p is the true token distribution and q* is the best distribution the
architecture class can express.  H(p) is architecture-independent.  Under
QSF, language interpretation is Born-rule measurement on a complex Hilbert
space, with joint distributions over multiple context-dependent measurements
exhibiting CHSH-style non-classical correlations.  A real-valued architecture
can only express classical (LHV) distributions and pays an irreducible KL
cost delta_classical for approximating the non-classical structure.  A
complex-valued architecture can in principle express the full quantum
distribution, paying no such cost:

    E_real - E_complex >= delta_classical(|S|, k)

where |S| is the empirical CHSH magnitude and k is the average number of
non-classical context-pair measurements per token.  This script computes
delta_classical from first principles under QSF, reporting only the gap
between architecture floors (the absolute floors require empirical H(p),
which is not available from first principles).

Method
------
1. Take the empirical |S| distribution from Agostino et al. 2026, Table I,
   aggregated as a single representative distribution by averaging the
   per-model summary statistics.
2. For each |S| in the support, parameterize a partial-entangled qubit pair
   |psi(theta)> = cos(theta)|00> + sin(theta)|11> with theta chosen to
   reproduce |S|.  Apply the Born rule with optimal CHSH measurement angles
   to get the quantum joint distribution P_Q(a,b|x,y).
3. Find the closest classical LHV approximation P_C* by minimizing KL
   divergence over the LHV polytope (convex hull of 16 deterministic
   strategies).
4. Per-pair gap:  delta_CHSH(|S|) = E_{x,y}[D_KL(P_Q || P_C*)]
5. Distribution-weighted per-pair gap:
       <delta_CHSH> = integral over |S| of delta_CHSH(|S|) * p_S(|S|) d|S|,
   truncated at the Tsirelson bound 2 sqrt(2) (super-Tsirelson observations
   are treated as artifact).
6. Per-token gap:  delta_classical = k * <delta_CHSH>, parameterized by k.

Usage:
    cd /Users/caug/npcww/qnlp/qllm-private
    uv run python v6/paper/compute_irreducible_loss.py
"""

import numpy as np
from itertools import product
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.integrate import quad


# ----------------------------------------------------------------------
# Quantum CHSH joint distribution and KL projection onto LHV polytope
# ----------------------------------------------------------------------

def quantum_joint_partial_entangled(theta):
    """For state |psi> = cos(theta)|00> + sin(theta)|11> with optimal
    CHSH measurement angles, return P_Q[x,y,a,b] for x,y,a,b in {0,1}.
    Optimal Bob Pauli angle: tan(mu) = sin(2 theta).  Theta = pi/4 gives
    the Bell state and saturates the Tsirelson bound 2 sqrt(2)."""
    mu = np.arctan2(np.sin(2 * theta), 1.0)
    angles_a = [0.0, np.pi / 2.0]
    angles_b = [mu, -mu]
    c, s = np.cos(theta), np.sin(theta)
    psi = np.array([c, 0.0, 0.0, s])

    def projector(phi, outcome):
        if outcome == 0:
            v = np.array([np.cos(phi / 2), np.sin(phi / 2)])
        else:
            v = np.array([-np.sin(phi / 2), np.cos(phi / 2)])
        return np.outer(v, v.conj())

    P = np.zeros((2, 2, 2, 2))
    for x in (0, 1):
        for y in (0, 1):
            for a in (0, 1):
                for b in (0, 1):
                    M = np.kron(projector(angles_a[x], a),
                                projector(angles_b[y], b))
                    P[x, y, a, b] = float(psi @ M @ psi)
            P[x, y] /= P[x, y].sum()
    return P


def chsh_value(P):
    def E(x, y):
        return sum(((-1) ** (a + b)) * P[x, y, a, b]
                   for a, b in product((0, 1), (0, 1)))
    return E(0, 0) + E(0, 1) + E(1, 0) - E(1, 1)


def theta_for_chsh(target_S):
    """Inverse of |S(theta)| = 2 sqrt(1 + sin^2(2 theta))."""
    if target_S <= 2.0:
        return 0.0
    target_S = min(target_S, 2 * np.sqrt(2) - 1e-6)
    val = (target_S / 2.0) ** 2 - 1.0
    val = max(0.0, min(val, 1.0))
    sin2theta = np.sqrt(val)
    return 0.5 * np.arcsin(sin2theta)


def lhv_strategies():
    strategies = []
    for ax0, ax1, by0, by1 in product((0, 1), repeat=4):
        S = np.zeros((2, 2, 2, 2))
        for x, y in product((0, 1), repeat=2):
            a = ax0 if x == 0 else ax1
            b = by0 if y == 0 else by1
            S[x, y, a, b] = 1.0
        strategies.append(S)
    return np.array(strategies)


def kl_to_best_lhv(P_quantum, eps=1e-10):
    """KL from P_quantum to its closest LHV approximation, averaged over (x,y)."""
    strategies = lhv_strategies()
    n_strat = len(strategies)

    def neg_log_likelihood(z):
        w = np.exp(z - z.max())
        w = w / w.sum()
        P_lhv = np.tensordot(w, strategies, axes=1)
        kl = 0.0
        for x, y in product((0, 1), repeat=2):
            p = P_quantum[x, y].flatten()
            q = P_lhv[x, y].flatten() + eps
            mask = p > 0
            kl += np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask])))
        return kl / 4.0

    rng = np.random.default_rng(0)
    best = np.inf
    for _ in range(20):
        z0 = rng.standard_normal(n_strat)
        res = minimize(neg_log_likelihood, z0, method='L-BFGS-B',
                       options={'maxiter': 500})
        if res.fun < best:
            best = res.fun
    return best


def delta_chsh_at_S(S):
    """Per-pair classical-approximation KL cost at CHSH magnitude |S|."""
    if S <= 2.0:
        return 0.0
    theta = theta_for_chsh(S)
    P_Q = quantum_joint_partial_entangled(theta)
    return kl_to_best_lhv(P_Q)


# ----------------------------------------------------------------------
# Empirical |S| distribution from Agostino et al. 2026, Table I
# ----------------------------------------------------------------------

# Per-model summary stats: (sigma, skew, kurtosis_excess, IQR, viol_pct)
# Mean is reported in the table footers/text as ~2.0 across models.
TABLE_I_STATS = [
    # (sigma, skew, kurt, IQR, viol%)
    (0.43, -1.27,  0.27, 0.35, 10.0),  # Claude Haiku 4.5
    (0.62, -0.97,  1.49, 0.55, 40.0),  # Claude Sonnet 4.6
    (0.38,  0.64,  6.48, 0.14, 29.2),  # Qwen3 0.6B
    (0.22,  0.24,  0.62, 0.20, 30.0),  # Qwen3 4B
    (0.47, -0.12,  0.47, 0.40, 55.6),  # Cogito 3B
    (0.64,  0.02,  0.38, 0.51, 45.0),  # Ministral 3B
    (0.31, -0.78,  2.27, 0.33, 34.5),  # Llama 3.2 3B
    (0.49,  0.34,  0.10, 0.62, 46.2),  # Gemma3 4B
    (0.48,  0.23,  0.50, 0.60, 54.4),  # Gemma3 12B
    (0.49, -0.27,  1.41, 0.55, 40.0),  # Gemma3 27B
    (0.28, -0.09, -0.10, 0.30, 25.9),  # GPT-4o Mini
    (0.46,  0.35,  0.06, 0.62, 40.0),  # GPT-OSS
    (0.59,  0.40, -0.15, 0.79, 37.0),  # Mistral Small 3.2
    (0.37,  0.68,  4.99, 0.40, 40.0),  # DeepSeek-V3
    (0.37, -1.32,  3.80, 0.29, 29.4),  # Gemini 2.5 Flash
    (0.45,  0.65,  0.44, 0.56, 34.6),  # Gemini 3 Flash
]


def aggregate_stats():
    arr = np.array(TABLE_I_STATS)
    means = arr.mean(axis=0)
    return {
        'mu_S':     2.0,            # paper text: every model modes at S ~ 2.0
        'sigma':    means[0],
        'skew':     means[1],
        'kurtosis': means[2],
        'iqr':      means[3],
        'viol_pct': means[4],
    }


def empirical_S_pdf(S, mu, sigma):
    """Approximate the aggregated |S| distribution as a normal centered at
    mu with width sigma.  The aggregated skew is near zero (~ -0.04), so a
    symmetric distribution is a defensible first-order approximation; the
    leptokurtosis (~1.6) and per-model skew are not captured but the
    integral is dominated by the bulk near |S| ~ 2 where these matter
    little for delta_CHSH."""
    return norm.pdf(S, loc=mu, scale=sigma)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("First-principles gap between E_real and E_complex floors")
    print("under QSF (Agostino et al. 2025, 2026)")
    print("=" * 70)
    print()

    stats = aggregate_stats()
    print("Aggregated |S| distribution from Agostino et al. 2026, Table I")
    print(f"  averaged across {len(TABLE_I_STATS)} models:")
    print(f"    mu (mode/mean of |S|)  = {stats['mu_S']:.2f}")
    print(f"    sigma                  = {stats['sigma']:.3f}")
    print(f"    skew (mean per-model)  = {stats['skew']:+.3f}")
    print(f"    excess kurtosis (mean) = {stats['kurtosis']:+.3f}")
    print(f"    IQR (mean per-model)   = {stats['iqr']:.3f}")
    print(f"    violation rate         = {stats['viol_pct']:.1f}%")
    print()
    print(f"Approximation: normal(mu={stats['mu_S']:.2f}, sigma={stats['sigma']:.3f}),")
    print(f"truncated to [2, 2 sqrt(2)] for non-classical regime.")
    print(f"(Super-Tsirelson observations treated as overfitting artifact.)")
    print()

    # Tabulate delta_CHSH(|S|) on a grid
    print("delta_CHSH(|S|) per measurement pair:")
    print(f"  {'|S|':>6}  {'delta_CHSH (nats)':>18}")
    grid = np.linspace(2.0, 2 * np.sqrt(2) - 1e-3, 11)
    delta_grid = []
    for S in grid:
        d = delta_chsh_at_S(S)
        delta_grid.append(d)
        print(f"  {S:6.3f}  {d:18.5f}")
    delta_grid = np.array(delta_grid)
    print()

    # Distribution-weighted integral
    S_lo, S_hi = 2.0, 2 * np.sqrt(2)
    pdf = lambda S: empirical_S_pdf(S, stats['mu_S'], stats['sigma'])
    norm_const, _ = quad(pdf, S_lo, S_hi)
    delta_interp = lambda S: np.interp(S, grid, delta_grid)
    weighted, _ = quad(lambda S: delta_interp(S) * pdf(S) / norm_const,
                       S_lo, S_hi)
    p_violation_in_range = norm_const   # fraction of mass with 2 < |S| < 2 sqrt(2)
    delta_unconditional = weighted * p_violation_in_range
    print(f"Mass of fitted distribution in (2, 2 sqrt(2)] = {p_violation_in_range:.3f}")
    print(f"<delta_CHSH | non-classical>  = {weighted:.5f} nats per pair")
    print(f"<delta_CHSH> (unconditional)  = {delta_unconditional:.5f} nats per pair")
    print()

    # Per-token cost
    print("Per-token classical-approximation cost = k * <delta_CHSH>")
    print("(k = average number of non-classical context-pair measurements per token)")
    for k in (1, 2, 5, 10):
        d_token = k * delta_unconditional
        print(f"  k = {k:2d}:  delta_classical = {d_token:.5f} nats per token")
    print()

    print("Predicted complex-valued irreducible-loss floor")
    print("-" * 50)
    print(" Anchoring at the Kaplan/Chinchilla real-valued floor estimate")
    print(" E_real_floor = 1.69 nats per token (computed from real-valued")
    print(" transformer fits on web-scale corpora), the QSF prediction is")
    print()
    print("     E_complex_floor <= 1.69 - k * <delta_CHSH>")
    print()
    print(f" with <delta_CHSH> = {delta_unconditional:.5f} nats per measurement pair")
    print(" (distribution-weighted over the empirical |S| from Agostino 2026,")
    print(" truncated at the Tsirelson bound).  Predicted floors:")
    print()
    print(f"  {'k':>3}  {'delta_classical':>15}  {'E_complex_floor':>17}  {'reduction':>10}")
    for k in (1, 2, 5, 10):
        d_token = k * delta_unconditional
        E_complex = 1.69 - d_token
        reduction_pct = 100 * d_token / 1.69
        print(f"  {k:3d}  {d_token:15.5f}  {E_complex:17.3f}  {reduction_pct:9.1f}%")
    print()
    print(" The QSF framework therefore predicts that a complex-valued language")
    print(" model should approach an asymptotic loss strictly below the 1.69-nat")
    print(" floor characterized for real-valued architectures, with the margin")
    print(" set by the empirical CHSH magnitude and per-token entanglement degree.")


if __name__ == "__main__":
    main()
