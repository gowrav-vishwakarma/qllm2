#!/usr/bin/env python3
"""Generate validation PPL vs epoch figure for medium-pam-v3 (from paper tables)."""
from pathlib import Path

import matplotlib.pyplot as plt

epochs = list(range(1, 11))
val_ppl = [57.94, 43.83, 38.69, 35.88, 33.82, 32.25, 31.22, 30.40, 30.01, 29.95]

out = Path(__file__).resolve().parent.parent / "figures" / "val_ppl_curve.pdf"
out.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "figure.figsize": (4.8, 2.8),
    }
)
fig, ax = plt.subplots()
ax.plot(epochs, val_ppl, color="#1a5276", marker="o", markersize=4, linewidth=1.2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation perplexity")
ax.set_xticks(epochs)
ax.set_xlim(0.5, 10.5)
ax.grid(True, alpha=0.35, linestyle="--")
ax.set_title("medium-pam-v3 (WikiText-103)")
fig.tight_layout()
fig.savefig(out, format="pdf", bbox_inches="tight")
print(f"Wrote {out}")
