"""Scaling analysis: log-log fit with uncertainty from late-epoch variance.

Uses last 3 end-of-epoch val PPLs + mid-epoch val PPLs (~6 samples per scale)
to get mean and stdev, then fits power law in log-log space with error bars.

Usage:
    cd /Users/caug/npcww/qnlp/qllm-private
    uv run python v6/paper/plot_scaling.py
"""

import os
import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matplotlibrc'))
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──
# For each scale and model: list of late-epoch val PPLs
# (end-of-epoch + mid-epoch checkpoints from the last few epochs)

# Methodology: last 3 epochs, mid-epoch val + end-of-epoch val = 6 samples per scale.
# For models with no mid-epoch logs (PyTorch transformer), 3 end-of-epoch only.

# 5M: epochs 8,9,10 — [mid, final] x3
qpam_5m = [89.19, 88.40, 87.67, 87.31, 87.06, 87.02]
rpam_5m = [61.36, 61.11, 60.84, 60.59, 60.53, 60.52]
trans_5m = [142.66, 140.82, 140.47]  # PyTorch: end-of-epoch only

# 10M: epochs 8,9,10 — [mid, final] x3
qpam_10m = [60.13, 59.48, 59.14, 58.83, 58.75, 58.71]
rpam_10m = [40.84, 40.52, 40.42, 40.23, 40.22, 40.20]
trans_10m = [76.87, 75.37, 74.11]  # epochs 7,8,9 end-of-epoch (ep10 still running)

# 25M: QPAM epochs 8,9,10 (checkpoint eval; no mid-epoch logs available)
# RPAM 25M — in progress
qpam_25m = [39.18, 38.85, 38.84]
rpam_25m = None  # TODO: fill after run

# 100M: QPAM epochs 8,9,10 (MLX v2, still improving); RPAM epochs 5,6,7 (last converging)
qpam_100m = [37.55, 36.46, 35.61, 34.63, 33.75, 33.19]
rpam_100m = [26.79, 26.03, 26.00, 25.59, 25.82, 25.42]
trans_100m = [27.63, 27.38, 27.10]  # end-of-epoch only

# Collect raw samples and means per (scale, model)
def stats(vals):
    return np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0

raw = {}   # (scale, model) -> list of PPL values
data = {}  # (scale, model) -> (mean, std, n)
for scale, model, vals in [
    (5, 'QPAM', qpam_5m), (5, 'RPAM', rpam_5m), (5, 'Trans', trans_5m),
    (10, 'QPAM', qpam_10m), (10, 'RPAM', rpam_10m), (10, 'Trans', trans_10m),
    (25, 'QPAM', qpam_25m),
    *([(25, 'RPAM', rpam_25m)] if rpam_25m is not None else []),
    (100, 'QPAM', qpam_100m), (100, 'RPAM', rpam_100m), (100, 'Trans', trans_100m),
]:
    raw[(scale, model)] = vals
    m, s = stats(vals)
    data[(scale, model)] = (m, s, len(vals))

# ── OLS fit on all individual samples in log-log space ──
# Each late-epoch PPL sample is one data point: (log10(N), log10(PPL))

models = ['QPAM', 'RPAM', 'Trans']
fits = {}
N_BOOT = 10000
rng = np.random.default_rng(42)

for model in models:
    log_p_all = []
    log_ppl_all = []
    for scale in [5, 10, 25, 100]:
        if (scale, model) in raw:
            for ppl in raw[(scale, model)]:
                log_p_all.append(np.log10(scale))
                log_ppl_all.append(np.log10(ppl))

    log_p_all = np.array(log_p_all)
    log_ppl_all = np.array(log_ppl_all)

    # OLS
    xm = log_p_all.mean(); ym = log_ppl_all.mean()
    b_yx = np.sum((log_p_all - xm) * (log_ppl_all - ym)) / np.sum((log_p_all - xm)**2)
    a_yx = ym - b_yx * xm

    # Bootstrap on samples
    boot_slopes = []
    boot_intercepts = []
    n = len(log_p_all)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        xb = log_p_all[idx]; yb = log_ppl_all[idx]
        xmb = xb.mean(); ymb = yb.mean()
        denom = np.sum((xb - xmb)**2)
        if denom == 0: continue
        bb = np.sum((xb - xmb) * (yb - ymb)) / denom
        ab = ymb - bb * xmb
        boot_slopes.append(bb)
        boot_intercepts.append(ab)

    slope_err = np.std(boot_slopes)

    # For plotting: aggregate means and stds per scale
    scales_plot = []
    means_plot = []
    sigmas_plot = []
    for scale in [5, 10, 25, 100]:
        if (scale, model) in data:
            m, s, nn = data[(scale, model)]
            scales_plot.append(scale)
            means_plot.append(m)
            sigmas_plot.append(max(s, 0.1))

    fits[model] = {
        'scales': np.array(scales_plot), 'means': np.array(means_plot),
        'sigmas': np.array(sigmas_plot),
        'coeffs': (b_yx, a_yx), 'slope_err': slope_err,
        'boot_slopes': boot_slopes, 'boot_intercepts': boot_intercepts,
    }

    print(f'{model}: slope = {b_yx:.4f} ± {slope_err:.4f}, '
          f'PPL = {10**a_yx:.1f} * params^({b_yx:.3f})')
    for s, m, sig in zip(scales_plot, means_plot, sigmas_plot):
        print(f'  {s:>3}M: {m:.2f} ± {sig:.2f}')

# ── Crossover analysis ──
print()
for m1, m2 in [('QPAM', 'RPAM'), ('QPAM', 'Trans'), ('RPAM', 'Trans')]:
    a1, b1 = fits[m1]['coeffs']
    a2, b2 = fits[m2]['coeffs']
    if abs(a1 - a2) > 0.001:
        log_p_cross = (b2 - b1) / (a1 - a2)
        p_cross = 10**log_p_cross
        ppl_cross = 10**(a1 * log_p_cross + b1)
        if 0.1 < p_cross < 1e8:
            if p_cross >= 1000:
                print(f'{m1}/{m2} crossover: ~{p_cross/1000:.1f}B params, PPL ~{ppl_cross:.1f}')
            else:
                print(f'{m1}/{m2} crossover: ~{p_cross:.0f}M params, PPL ~{ppl_cross:.1f}')
        else:
            print(f'{m1}/{m2}: no crossover in reasonable range')
    else:
        print(f'{m1}/{m2}: parallel slopes, no crossover')

# ── Plot ──
fig, ax = plt.subplots(figsize=(6, 4.5))

colors = {'QPAM': '#2563eb', 'RPAM': '#dc2626', 'Trans': '#16a34a'}
markers = {'QPAM': 'o', 'RPAM': 's', 'Trans': '^'}
labels = {'QPAM': 'QPAM (complex)', 'RPAM': 'RPAM (real)', 'Trans': 'Transformer'}

p_fine = np.logspace(np.log10(3), np.log10(1000000), 200)

for model in models:
    f = fits[model]
    a, b = f['coeffs']
    c = colors[model]

    # Individual samples
    first = True
    for scale in [5, 10, 25, 100]:
        if (scale, model) in raw:
            vals = raw[(scale, model)]
            ax.scatter([scale]*len(vals), vals, marker=markers[model],
                       color=c, s=30, zorder=5, alpha=0.7,
                       label=labels[model] if first else None)
            first = False

    # Trend line
    ax.plot(p_fine, 10**(a * np.log10(p_fine) + b), '--', color=c, alpha=0.4, lw=1.2)

# Crossover annotations: place text inside plot bounds
annot_cfg = {
    # (m1, m2): (x_text_mult, y_text_abs)  — y_text_abs in PPL units
    ('QPAM', 'RPAM'):  (0.35, 55),
    ('QPAM', 'Trans'): (3.5,  55),
}
for m1, m2 in [('QPAM', 'RPAM')]:
    a1, b1 = fits[m1]['coeffs']
    a2, b2 = fits[m2]['coeffs']
    if abs(a1 - a2) > 0.001:
        log_p_cross = (b2 - b1) / (a1 - a2)
        p_cross = 10**log_p_cross
        ppl_cross = 10**(a1 * log_p_cross + b1)
        if 1 < p_cross < 1e8:
            ax.axvline(p_cross, color='#555', ls=':', alpha=0.4, lw=0.8)
            label = f'{p_cross/1000:.1f}B' if p_cross >= 1000 else f'{p_cross:.0f}M'
            xm, yt = annot_cfg.get((m1, m2), (1.8, 40))
            ax.annotate(f'{m1}/{m2}\n{label}',
                        xy=(p_cross, max(ppl_cross, 20)),
                        xytext=(p_cross * xm, yt),
                        fontsize=8, ha='center', color='#555',
                        arrowprops=dict(arrowstyle='->', color='#555', lw=0.6))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Parameters (M)')
ax.set_ylabel('Validation PPL')
ax.set_xlim(3, 1000000)
ax.set_ylim(15, 300)
ax.legend(fontsize=9, loc='upper right')

fig.tight_layout()
out = '/Users/caug/npcww/qnlp/qllm-private/v6/paper/figures/scaling_loglog.pdf'
fig.savefig(out)
fig.savefig(out.replace('.pdf', '.png'), dpi=150)
print(f'\nSaved {out}')