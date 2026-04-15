"""Scaling analysis: log-log fit with uncertainty from late-epoch variance.

Uses last 3 end-of-epoch val PPLs + mid-epoch val PPLs (~6 samples per scale)
to get mean and stdev, then fits power law in log-log space with error bars.

Usage:
    cd /Users/caug/npcww/qnlp/qllm-private
    uv run python v6/paper/plot_scaling.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Data ──
# For each scale and model: list of late-epoch val PPLs
# (end-of-epoch + mid-epoch checkpoints from the last few epochs)

# Methodology: last 3 epochs, mid-epoch val + end-of-epoch val = 6 samples per scale.
# For models with no mid-epoch logs (PyTorch transformer), 3 end-of-epoch only.

# 5M: epochs 8,9,10 — [mid, final] x3
qpam_5m = [89.19, 88.72, 87.67, 87.35, 87.06, 87.02]
rpam_5m = [61.36, 61.11, 60.84, 60.59, 60.53, 60.52]
trans_5m = [142.66, 140.82, 140.47]  # PyTorch: end-of-epoch only

# 10M: QPAM epochs 4,5,6 (still training); RPAM epochs 8,9,10
qpam_10m = [72.85, 70.48, 67.46, 65.71, 64.42, 62.82]
rpam_10m = [40.84, 40.56, 40.42, 40.25, 40.22, 40.20]

# 100M: QPAM epochs 8,9,10 (MLX v2, still improving); RPAM epochs 5,6,7 (last converging)
qpam_100m = [37.55, 36.46, 35.61, 34.63, 33.75, 33.19]
rpam_100m = [26.79, 26.03, 26.00, 25.59, 25.82, 25.42]
trans_100m = [27.63, 27.38, 27.10]  # end-of-epoch only

# Compute mean and stdev
def stats(vals):
    return np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0

data = {}
for scale, model, vals in [
    (5, 'QPAM', qpam_5m), (5, 'RPAM', rpam_5m), (5, 'Trans', trans_5m),
    (10, 'QPAM', qpam_10m), (10, 'RPAM', rpam_10m),
    (100, 'QPAM', qpam_100m), (100, 'RPAM', rpam_100m), (100, 'Trans', trans_100m),
]:
    m, s = stats(vals)
    data[(scale, model)] = (m, s, len(vals))

# ── BCES Fit ──
# Use BCES(Y|X) with bootstrap for uncertainty on crossover

models = ['QPAM', 'RPAM', 'Trans']
fits = {}
N_BOOT = 10000

rng = np.random.default_rng(42)

for model in models:
    scales = []
    means = []
    sigmas = []
    for scale in [5, 10, 100]:
        if (scale, model) in data:
            m, s, n = data[(scale, model)]
            scales.append(scale)
            means.append(m)
            sigmas.append(max(s, 0.1))

    scales = np.array(scales, dtype=float)
    means = np.array(means)
    sigmas = np.array(sigmas)

    log_p = np.log10(scales)
    log_ppl = np.log10(means)
    log_sigma = sigmas / (means * np.log(10))

    # BCES(Y|X) weighted fit
    w = 1.0 / log_sigma**2
    wx = np.sum(w * log_p) / np.sum(w)
    wy = np.sum(w * log_ppl) / np.sum(w)
    sxx = np.sum(w * (log_p - wx)**2) / np.sum(w)
    sxy = np.sum(w * (log_p - wx) * (log_ppl - wy)) / np.sum(w)
    b_yx = sxy / sxx
    a_yx = wy - b_yx * wx

    # Bootstrap for slope uncertainty
    boot_slopes = []
    boot_intercepts = []
    for _ in range(N_BOOT):
        # Resample y from Gaussian around measured values
        y_boot = log_ppl + rng.normal(0, log_sigma)
        w_b = 1.0 / log_sigma**2
        wx_b = np.sum(w_b * log_p) / np.sum(w_b)
        wy_b = np.sum(w_b * y_boot) / np.sum(w_b)
        sxx_b = np.sum(w_b * (log_p - wx_b)**2) / np.sum(w_b)
        sxy_b = np.sum(w_b * (log_p - wx_b) * (y_boot - wy_b)) / np.sum(w_b)
        b_b = sxy_b / sxx_b
        a_b = wy_b - b_b * wx_b
        boot_slopes.append(b_b)
        boot_intercepts.append(a_b)

    slope_err = np.std(boot_slopes)
    int_err = np.std(boot_intercepts)

    fits[model] = {
        'scales': scales, 'means': means, 'sigmas': sigmas,
        'log_p': log_p, 'log_ppl': log_ppl,
        'coeffs': (b_yx, a_yx), 'slope_err': slope_err, 'int_err': int_err,
        'boot_slopes': boot_slopes, 'boot_intercepts': boot_intercepts,
    }

    print(f'{model}: slope = {b_yx:.4f} ± {slope_err:.4f}, '
          f'PPL = {10**a_yx:.1f} * params^({b_yx:.3f})')
    for s, m, sig in zip(scales, means, sigmas):
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

p_fine = np.logspace(np.log10(3), np.log10(100000), 200)

for model in models:
    f = fits[model]
    a, b = f['coeffs']
    c = colors[model]

    # Data points with error bars
    ax.errorbar(f['scales'], f['means'], yerr=f['sigmas'],
                fmt=markers[model], color=c, ms=7, capsize=3, zorder=5,
                label=labels[model])

    # Trend line
    ax.plot(p_fine, 10**(a * np.log10(p_fine) + b), '--', color=c, alpha=0.4, lw=1.2)

# Crossover annotations
for m1, m2 in [('QPAM', 'Trans'), ('QPAM', 'RPAM')]:
    a1, b1 = fits[m1]['coeffs']
    a2, b2 = fits[m2]['coeffs']
    if abs(a1 - a2) > 0.001:
        log_p_cross = (b2 - b1) / (a1 - a2)
        p_cross = 10**log_p_cross
        ppl_cross = 10**(a1 * log_p_cross + b1)
        if 1 < p_cross < 1e7:
            ax.axvline(p_cross, color='#555', ls=':', alpha=0.4, lw=0.8)
            if p_cross >= 1000:
                label = f'{p_cross/1000:.0f}B'
            else:
                label = f'{p_cross:.0f}M'
            ax.annotate(f'{m1}/{m2}\n{label}', xy=(p_cross, ppl_cross),
                        xytext=(p_cross*1.8, ppl_cross*1.5),
                        fontsize=8, ha='center', color='#555',
                        arrowprops=dict(arrowstyle='->', color='#555', lw=0.6))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Parameters (M)')
ax.set_ylabel('Validation PPL')
ax.set_xlim(3, 100000)
ax.set_ylim(15, 300)
ax.legend(fontsize=9, loc='upper right')

fig.tight_layout()
out = '/Users/caug/npcww/qnlp/qllm-private/v6/paper/figures/scaling_loglog.pdf'
fig.savefig(out)
fig.savefig(out.replace('.pdf', '.png'), dpi=150)
print(f'\nSaved {out}')