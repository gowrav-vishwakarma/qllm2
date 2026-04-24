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

# Production sweep: lr=3e-05, batch=8, warmup=500, seq_len=512, epochs=10
# (configs in scripts/scaling_commands.txt). QPAM 10M omitted — restart in progress.
# Methodology: last 3 epochs, [ep8_end, ep8_mid_best, ep9_end, ep9_mid_best, ep10_end]

# 5M QPAM: canonical-config run completed Apr 22 (lr=3e-05, batch=8)
# [ep8_end, ep8_mid_best, ep9_end, ep9_mid_best, ep10_end]
qpam_5m = [259.7707, 259.7964, 258.3462, 258.3010, 258.1207]

# 5M RPAM: canonical-config run completed Apr 21 (lr=3e-05, batch=8)
# [ep8_end, ep8_mid_best, ep9_end, ep9_mid_best, ep10_end]
rpam_5m = [119.3802, 119.4058, 118.7168, 118.6796, 118.5472]

# 10M RPAM: canonical-config run completed Apr 21 (lr=3e-05, batch=8)
rpam_10m = [70.1365, 70.1186, 69.6156, 69.6337, 69.5743]

# 10M QPAM: canonical-config run completed Apr 22 (lr=3e-05, batch=8)
# [ep8_end, ep8_mid_best, ep9_end, ep9_mid_best, ep10_end]
qpam_10m = [133.7768, 133.9375, 127.6733, 127.6152, 123.4690]

# 25M (from logs/v6/scaling_sweep/*_25m_val_ppl.log — pulled from remote)
qpam_25m = [75.7337, 75.6768, 74.9918, 75.0107, 74.9205]
rpam_25m = [41.0135, 40.9486, 40.7149, 40.6964, 40.6845]

# 50M QPAM (from logs/v6/scaling_sweep/qpam_50m_val_ppl.log — pulled from remote)
# [ep8_end, ep8_mid_best, ep9_end, ep9_mid_best, ep10_end]
qpam_50m = [49.7379, 49.6918, 47.4510, 47.5868, 45.7015]

# 50M RPAM (from logs/v6/scaling_sweep/rpam_50m_val_ppl.log — pulled from remote)
rpam_50m = [35.8662, 35.8502, 35.7049, 35.7079, 35.6986]

# 100M: QPAM epochs 8-10 (6 samples incl. mid-epoch); RPAM epochs 5-7
qpam_100m = [37.55, 36.46, 35.61, 34.63, 33.75, 33.19]
rpam_100m = [26.79, 26.03, 26.00, 25.59, 25.82, 25.42]

# Collect raw samples and means per (scale, model)
def stats(vals):
    return np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0

raw = {}   # (scale, model) -> list of PPL values
data = {}  # (scale, model) -> (mean, std, n)
for scale, model, vals in [
    (5,   'QPAM', qpam_5m),
    (10,  'QPAM', qpam_10m),
    (25,  'QPAM', qpam_25m),
    (50,  'QPAM', qpam_50m),
    (100, 'QPAM', qpam_100m),
    (5,   'RPAM', rpam_5m),
    (10,  'RPAM', rpam_10m),
    (25,  'RPAM', rpam_25m),
    (50,  'RPAM', rpam_50m),
    (100, 'RPAM', rpam_100m),
]:
    raw[(scale, model)] = vals
    m, s = stats(vals)
    data[(scale, model)] = (m, s, len(vals))

# ── OLS fits in both log10(PPL) and log10(loss) spaces ──
# loss = ln(PPL); Kaplan/Hoffmann-style scaling laws fit power law in loss.

models = ['QPAM', 'RPAM']
N_BOOT = 10000
rng = np.random.default_rng(42)


def fit_loglog(model, transform):
    """OLS slope of log10(transform(ppl)) vs log10(N) on individual samples."""
    log_p_all, log_y_all = [], []
    for scale in [5, 10, 25, 50, 100]:
        if (scale, model) in raw:
            for ppl in raw[(scale, model)]:
                log_p_all.append(np.log10(scale))
                log_y_all.append(np.log10(transform(ppl)))
    log_p_all = np.array(log_p_all)
    log_y_all = np.array(log_y_all)
    xm = log_p_all.mean(); ym = log_y_all.mean()
    b = np.sum((log_p_all - xm) * (log_y_all - ym)) / np.sum((log_p_all - xm)**2)
    a = ym - b * xm
    return (b, a)


fits_ppl = {m: fit_loglog(m, lambda x: x) for m in models}
fits_loss = {m: fit_loglog(m, np.log) for m in models}

for model in models:
    b_p, a_p = fits_ppl[model]
    b_l, a_l = fits_loss[model]
    print(f'{model}: PPL slope = {b_p:.3f} (PPL = {10**a_p:.1f} * N^{b_p:.3f}); '
          f'loss slope = {b_l:.3f} (alpha-Kaplan-style)')

# ── Crossover (under PPL-space linear fit; loss-space fit gives a different N) ──
print()
b1, a1 = fits_ppl['QPAM']
b2, a2 = fits_ppl['RPAM']
if abs(b1 - b2) > 0.001:
    log_p_cross_ppl = (a2 - a1) / (b1 - b2)
    p_cross_ppl = 10**log_p_cross_ppl
    print(f'PPL-space fit crossover: ~{p_cross_ppl:.0f}M params')
b1, a1 = fits_loss['QPAM']
b2, a2 = fits_loss['RPAM']
if abs(b1 - b2) > 0.001:
    log_p_cross_loss = (a2 - a1) / (b1 - b2)
    p_cross_loss = 10**log_p_cross_loss
    print(f'Loss-space fit crossover: ~{p_cross_loss:.0f}M params')

# ── Plot: two panels (loss top, PPL bottom) sharing x-axis ──
fig, (ax_loss, ax_ppl) = plt.subplots(2, 1, figsize=(5, 8), sharex=True,
                                      gridspec_kw={'hspace': 0})

markers = {'QPAM': '^', 'RPAM': '*'}
labels = {'QPAM': 'QPAM (complex)', 'RPAM': 'RPAM (real)'}

log_p_fine = np.linspace(0, 4, 200)


colors = {'QPAM': '#2563eb', 'RPAM': '#dc2626'}


def draw_panel(ax, fits_dict, transform, ylabel, ylim):
    """Plot fits + error-bar points for transform(PPL)."""
    for model in models:
        b, a = fits_dict[model]
        c = colors[model]
        first = True
        for scale in [5, 10, 25, 50, 100]:
            if (scale, model) in raw:
                vals = transform(np.array(raw[(scale, model)]))
                mu = vals.mean()
                sigma = vals.std(ddof=1) if len(vals) > 1 else 0.0
                mu_y = np.log10(mu)
                sigma_y = sigma / (mu * np.log(10))
                lbl = f'{labels[model]}, slope = {b:.2f}' if first else None
                ax.errorbar(np.log10(scale), mu_y, yerr=sigma_y,
                            fmt=markers[model], color='black', markersize=2,
                            ecolor='black', elinewidth=0.8, capsize=4,
                            capthick=0.8, zorder=5, label=lbl)
                first = False
        ax.plot(log_p_fine, b * log_p_fine + a, '--', color=c, alpha=0.5, lw=1.2)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)


draw_panel(ax_loss, fits_loss, np.log,
           r'$\log_{10}$ Validation Loss', (0.3, 0.9))
draw_panel(ax_ppl, fits_ppl, lambda x: x,
           r'$\log_{10}$ Validation PPL', (0, 3))

def cross_label(p_cross):
    return (rf'$\sim${p_cross/1000:.1f}B' if p_cross >= 1000
            else rf'$\sim${round(p_cross / 10) * 10:.0f}M')


# Crossover markers per panel (different fits → different N)
b1, a1 = fits_ppl['QPAM']; b2, a2 = fits_ppl['RPAM']
if abs(b1 - b2) > 0.001:
    log_p_cross = (a2 - a1) / (b1 - b2)
    log_ppl_cross = b1 * log_p_cross + a1
    if 0 <= log_p_cross <= 4:
        ax_ppl.axvline(log_p_cross, color='#555', ls=':', alpha=0.4, lw=0.8)
        ax_ppl.annotate(f'crossover\n{cross_label(10**log_p_cross)}',
                        xy=(log_p_cross, log_ppl_cross),
                        xytext=(log_p_cross + 0.6, log_ppl_cross + 0.6),
                        fontsize=8, ha='center', color='#555',
                        arrowprops=dict(arrowstyle='->', color='#555', lw=0.6))

b1, a1 = fits_loss['QPAM']; b2, a2 = fits_loss['RPAM']
if abs(b1 - b2) > 0.001:
    log_p_cross = (a2 - a1) / (b1 - b2)
    log_loss_cross = b1 * log_p_cross + a1
    if 0 <= log_p_cross <= 4:
        ax_loss.axvline(log_p_cross, color='#555', ls=':', alpha=0.4, lw=0.8)
        ax_loss.annotate(f'crossover\n{cross_label(10**log_p_cross)}',
                         xy=(log_p_cross, log_loss_cross),
                         xytext=(log_p_cross - 0.5, log_loss_cross + 0.12),
                         fontsize=8, ha='center', color='#555',
                         arrowprops=dict(arrowstyle='->', color='#555', lw=0.6))

ax_ppl.set_xlabel(r'$\log_{10}$ Parameters (M)')
ax_ppl.set_xlim(0, 4)
ax_ppl.set_xticks([0, 1, 2, 3, 4])
ax_ppl.set_xticks([0.5, 1.5, 2.5, 3.5], minor=True)

ax_loss.legend(fontsize=9, loc=3, frameon=False)
ax_ppl.legend(fontsize=9, loc='lower left', frameon=False,
              bbox_to_anchor=(0.02, 0.05),
              bbox_transform=ax_ppl.get_yaxis_transform())
ax_ppl.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])

fig.tight_layout()
out = '/Users/caug/npcww/qnlp/qllm-private/v6/paper/figures/scaling_loglog.pdf'
fig.savefig(out)
fig.savefig(out.replace('.pdf', '.png'), dpi=400)
print(f'\nSaved {out}')