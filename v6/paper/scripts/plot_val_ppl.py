#!/usr/bin/env python3
"""Generate val-PPL comparison figure (interleaved PAM, sequential PAM, transformer)."""
from pathlib import Path

import matplotlib.pyplot as plt

epochs = list(range(1, 11))

pam_interleaved = [57.94, 43.83, 38.69, 35.88, 33.82, 32.25, 31.22, 30.40, 30.01, 30.0]
pam_sequential = [None, None, None, 47.19, 43.55, 41.43, 40.11, 39.34, 39.02, 38.95]
transformer = [53.74, 39.42, 34.76, 31.96, 30.39, 29.02, 28.09, 27.46, 27.15, 27.1]

fig_dir = Path(__file__).resolve().parent.parent / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "figure.figsize": (4.8, 3.0),
    }
)
fig, ax = plt.subplots()

ax.plot(epochs, pam_interleaved, color="#1a5276", marker="o", markersize=4,
        linewidth=1.2, label="PAM interleaved (this work)")
seq_epochs = [e for e, v in zip(epochs, pam_sequential) if v is not None]
seq_vals = [v for v in pam_sequential if v is not None]
ax.plot(seq_epochs, seq_vals, color="#7d3c98", marker="s", markersize=4,
        linewidth=1.2, linestyle="--", label="PAM sequential")
ax.plot(epochs, transformer, color="#b03a2e", marker="^", markersize=4,
        linewidth=1.2, label="Transformer (matched)")

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation perplexity")
ax.set_xticks(epochs)
ax.set_xlim(0.5, 10.5)
ax.grid(True, alpha=0.35, linestyle="--")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()

for name in ("val_ppl_comparison.pdf", "val_ppl_curve.pdf"):
    out = fig_dir / name
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"Wrote {out}")
