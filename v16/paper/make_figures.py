"""Draw every figure of the paper from data/numbers.json (python3 make_figures.py; writes figures/*.pdf and *.png)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data", "numbers.json")))
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
COL = {"PAM-2": "#1565c0", "Transformer++": "#9e9e9e", "Mamba-2": "#ef6c00", "PAM 16nb": "#64b5f6", "PAM-2 (M2)": "#1565c0"}
SHORT = {"PAM-2": "PAM-2", "Mamba-2": "Mamba-2", "Transformer++": "T++"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout(rect=(0, 0, 1, 0.93) if fig._suptitle else None)
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)


def fig1_control():
    """Headline: the equal-data control at the SCI-1 recipe."""
    c = D["ctl_1B"]["bodies"]; names = ["PAM-2", "Mamba-2", "Transformer++"]
    fig, ax = plt.subplots(1, 4, figsize=(9.2, 2.5))
    metrics = [("s8_D", "Rule induced from examples\n(s8 D, chance 0.125)"), ("told_trained_2k", "Rule told in prose,\nheld in state (2k)"),
               (None, "Fact recall past trained length\n(8k / 16k)"), ("holdout", "Held-out loss (nats, lower better)")]
    for i, (k, title) in enumerate(metrics):
        a = ax[i]
        if k == "holdout":
            v = [c[n]["holdout"] for n in names]; a.bar([SHORT[n] for n in names], v, color=[COL[n] for n in names]); a.set_ylim(3.30, 3.48)
            for j, x in enumerate(v): a.text(j, x + 0.003, f"{x:.3f}", ha="center", fontsize=8)
        elif k is None:
            w = 0.38; x = np.arange(3)
            for s, (lab, idx) in enumerate([("8k", 2), ("16k", 3)]):
                v = [c[n]["needle"][idx] for n in names]
                a.bar(x + (s - 0.5) * w, v, w, color=[COL[n] for n in names], alpha=1 if s == 0 else 0.55, label=lab)
                for j, y in enumerate(v): a.text(x[j] + (s - 0.5) * w, y + 0.01, f"{y:.2f}", ha="center", fontsize=7)
            a.set_xticks(x); a.set_xticklabels([SHORT[n] for n in names]); a.set_ylim(0, 1); a.legend(frameon=False, fontsize=7)
        else:
            v = [c[n][k] for n in names]; err = [c[n].get(k + "_se", 0) for n in names]
            a.bar([SHORT[n] for n in names], v, yerr=err, color=[COL[n] for n in names], capsize=3); a.set_ylim(0, 1)
            for j, x in enumerate(v): a.text(j, x + 0.06, f"{x:.2f}", ha="center", fontsize=8)
            if k == "s8_D": a.axhline(0.125, ls=":", c="k", lw=0.8)
        a.set_title(title, fontsize=8.5); a.tick_params(axis="x", labelsize=7.5)
    fig.suptitle("Three bodies, ~103M params each, 1.0B tokens of the identical data (SEQ 4096)", fontsize=9)
    save(fig, "fig1_control")


def fig2_distance():
    c = D["ctl_1B"]; names = ["PAM-2", "Mamba-2", "Transformer++"]
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    for n in names:
        ax[0].plot(c["s8_L_labels"], c["bodies"][n]["s8_by_L"], "o-", c=COL[n], label=n)
        ax[1].plot(c["needle_L_labels"], c["bodies"][n]["needle"], "o-", c=COL[n], label=n)
    ax[0].axhline(0.125, ls=":", c="k", lw=0.8); ax[0].set_ylim(0, 1); ax[1].set_ylim(0, 1)
    ax[0].set_xlabel("filler between the 8 examples and the question (tokens)"); ax[0].set_ylabel("rule applied to a new case")
    ax[0].set_title("Carrying an induced rule across text", fontsize=9)
    ax[1].axvline(1.5, ls="--", c="k", lw=0.8); ax[1].text(1.55, 0.95, "trained length 4k", fontsize=7)
    ax[1].set_xlabel("distance of the fact back (tokens)"); ax[1].set_ylabel("recall (20-way)")
    ax[1].set_title("Recalling a fact inside and beyond the trained length", fontsize=9)
    ax[0].legend(frameon=False, fontsize=7.5)
    save(fig, "fig2_distance")


def fig3_tiers():
    """Rules-at-distance and language across the equal-data tiers."""
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    t6, t15, ctl = D["tier_600M"], D["tier_1p5B"], D["ctl_1B"]["bodies"]
    tiers = ["600M\n(general mix)", "1.5B\n(general mix)", "1.0B\n(SCI-1 mix)"]
    rows = {"PAM-2": [np.mean(t6["s8_D"]["PAM-2"]), t15["s8_D"]["PAM-2 (M2)"], ctl["PAM-2"]["s8_D"]],
            "Transformer++": [np.mean(t6["s8_D"]["Transformer++"]), t15["s8_D"]["Transformer++"], ctl["Transformer++"]["s8_D"]],
            "Mamba-2": [np.mean(t6["s8_D"]["Mamba-2"]), t15["s8_D"]["Mamba-2"], ctl["Mamba-2"]["s8_D"]]}
    x = np.arange(3); w = 0.26
    for i, (n, v) in enumerate(rows.items()):
        ax[0].bar(x + (i - 1) * w, v, w, color=COL[n], label=n)
    ax[0].axhline(0.125, ls=":", c="k", lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels(tiers, fontsize=7.5)
    ax[0].set_ylim(0, 0.85); ax[0].set_ylabel("s8 D (rule at 2k-4k distance)"); ax[0].set_title("Rules carried in state, by tokens tier", fontsize=9)
    ax[0].legend(frameon=False, fontsize=7)
    hold = {"PAM-2": [t6["holdout"]["PAM-2"], t15["holdout"]["PAM-2 (M2)"], ctl["PAM-2"]["holdout"]],
            "Transformer++": [t6["holdout"]["Transformer++"], t15["holdout"]["Transformer++"], ctl["Transformer++"]["holdout"]],
            "Mamba-2": [t6["holdout"]["Mamba-2"], t15["holdout"]["Mamba-2"], ctl["Mamba-2"]["holdout"]]}
    for i, (n, v) in enumerate(hold.items()):
        ax[1].bar(x + (i - 1) * w, v, w, color=COL[n], label=n)
        for j, y in enumerate(v): ax[1].text(x[j] + (i - 1) * w, y + 0.004, f"{y:.3f}", ha="center", fontsize=5.5, rotation=90)
    ax[1].set_ylim(3.25, 3.62); ax[1].set_xticks(x); ax[1].set_xticklabels(tiers, fontsize=7.5)
    ax[1].set_ylabel("held-out loss (nats)"); ax[1].set_title("Language, same tiers (lower is better)", fontsize=9)
    save(fig, "fig3_tiers")


def fig4_sci1():
    s = D["sci1"]; x = s["tokens_B"]
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 2.6))
    ax[0].plot(x, s["s8_D"], "o-", c="#1565c0", label="induced rule (s8 D)")
    ax[0].plot(x, s["told_2k"], "s-", c="#2e7d32", label="told rule held, 2k (twin read)")
    ax[0].set_ylim(0, 1); ax[0].set_xlabel("tokens trained (B)"); ax[0].set_title("Two rule skills over training", fontsize=9)
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")
    for b, lab in [(4.0, "rule docs cut"), (5.0, "restored"), (7.0, "value-copy out")]:
        ax[0].axvline(b, ls=":", c="k", lw=0.7); ax[0].text(b + 0.05, 0.03, lab, fontsize=6, rotation=90)
    f = lambda k: [np.nan if v is None else v for v in s[k]]
    ax[1].plot(x, f("skill_8k"), "o-", label="skill pack @8k"); ax[1].plot(x, f("clk2_8k"), "s-", label="one read used 8k later")
    ax[1].plot(x, f("needle_16k"), "^-", label="fact recall @16k"); ax[1].set_ylim(0, 1); ax[1].set_xlabel("tokens trained (B)")
    ax[1].set_title("Beyond the 4k trained length", fontsize=9); ax[1].legend(frameon=False, fontsize=7)
    ax[2].plot(x, s["hellaswag"], "o-", label="HellaSwag"); ax[2].plot(x, s["arc_e"], "s-", label="ARC-e"); ax[2].plot(x, s["piqa"], "^-", label="PIQA")
    ax2 = ax[2].twinx(); ax2.plot(x, f("holdout"), "k--", lw=1, label="held-out loss"); ax2.set_ylabel("loss (nats)")
    ax[2].set_xlabel("tokens trained (B)"); ax[2].set_title("Language and knowledge", fontsize=9)
    h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[2].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="center right")
    fig.suptitle("SCI-1: PAM-2 386M, milestones every 1B tokens", fontsize=9)
    save(fig, "fig4_sci1")


def fig5_memory():
    m = D["package"]["memory_128k_MB"]
    labels = ["PAM-2 + library\n(state + rows + text + organ)", "Mamba-2 state", "Transformer++ KV cache"]
    vals = [m["PAM-2 state"] + m["rows"] + m["text"] + m["organ"], m["Mamba-2 state"], m["Transformer++ KV"]]
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.barh(labels, vals, color=["#1565c0", "#ef6c00", "#9e9e9e"]); ax.set_xscale("log"); ax.set_xlabel("memory held per live session at 128k tokens (MB, log)")
    for i, v in enumerate(vals): ax.text(v * 1.15, i, f"{v:,.1f} MB", va="center", fontsize=8)
    ax.set_xlim(1, 2e4); save(fig, "fig5_memory")


def fig6_babilong():
    b = D["package"]["babilong"]; x = np.arange(5); w = 0.2
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.bar(x - 1.5 * w, b["tpp_asis"], w, color=COL["Transformer++"], label="Transformer++ as is")
    ax.bar(x - 0.5 * w, b["pam_asis"], w, color="#90caf9", label="PAM-2 as is")
    ax.bar(x + 0.5 * w, b["pam_package_4k"], w, color="#1565c0", label="PAM-2 + library, 4k")
    ax.bar(x + 1.5 * w, b["pam_package_32k"], w, color="#0d47a1", label="PAM-2 + library, 32k")
    ax.set_xticks(x); ax.set_xticklabels([t + {"qa1": "\nsingle fact", "qa2": "\ntwo facts", "qa3": "\nthree facts", "qa4": "\n2-arg relation", "qa5": "\n3-arg relation"}[t] for t in b["tasks"]], fontsize=7.5)
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy"); ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("BABILong (103M params, 600M tokens; no tuning on the benchmark)", fontsize=9)
    save(fig, "fig6_babilong")


def fig7_pythia():
    r = D["pythia_ref"]["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.5))
    py = [q for q in r if q["model"].startswith("Pythia")]; ours = [q for q in r if not q["model"].startswith("Pythia")]
    for a, k, t in [(ax[0], "arce", "ARC-easy"), (ax[1], "s8_D", "rule from examples (s8 D)")]:
        a.plot([q["tokens_B"] for q in py], [q[k] for q in py], "o-", c="#9e9e9e", label="Pythia-410M (the Pile)")
        a.plot([q["tokens_B"] for q in ours], [q[k] for q in ours], "*", ms=12, c="#1565c0", label="SCI-1 PAM-2 386M")
        a.set_xscale("log"); a.set_xlabel("tokens trained (B, log)"); a.set_title(t, fontsize=9)
    ax[1].axhline(0.125, ls=":", c="k", lw=0.8); ax[1].set_ylim(0, 0.8); ax[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Reference block: different data, no claim", fontsize=9)
    save(fig, "fig7_pythia")


def fig8_knockouts():
    k = D["knockouts_ctl_pam"]["arms"]; names = list(k); v = [k[n]["s8_D"] for n in names]
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    cols = ["#1565c0", "#c62828", "#c62828", "#ef6c00", "#9e9e9e", "#9e9e9e"]
    ax.barh(names[::-1], v[::-1], color=cols[::-1]); ax.axvline(0.125, ls=":", c="k", lw=0.8); ax.set_xlim(0, 0.7)
    for i, x in enumerate(v[::-1]): ax.text(x + 0.01, i, f"{x:.2f}", va="center", fontsize=8)
    ax.set_xlabel("s8 D, rule carried across filler (dotted: chance)"); ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title("Knockouts on the trained control model\n(one part switched off during the filler only)", fontsize=8.5)
    save(fig, "fig8_knockouts")


def fig9_package():
    c = D["pkg_suite_m9"]["cells"]; names = list(c)
    bare = [c[n]["bare"] for n in names]; pk = [c[n]["package"] for n in names]
    y = np.arange(len(names)); h = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.barh(y + h / 2, bare, h, color="#90caf9", label="model alone")
    ax.barh(y - h / 2, pk, h, color="#1565c0", label="model + library")
    for i, (b, p) in enumerate(zip(bare, pk)):
        ax.text(p + 0.01, i - h / 2, f"{p:.2f}", va="center", fontsize=7); ax.text(b + 0.01, i + h / 2, f"{b:.2f}", va="center", fontsize=7, color="#555")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5); ax.invert_yaxis(); ax.set_xlim(0, 1.1)
    ax.set_xlabel("accuracy (100 documents per cell; recall 20 per task)"); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("The same checkpoint, alone and with the library (386M, 9B tokens)", fontsize=8.5)
    save(fig, "fig9_package")


if __name__ == "__main__":
    for f in (fig1_control, fig2_distance, fig3_tiers, fig4_sci1, fig5_memory, fig6_babilong, fig7_pythia, fig8_knockouts, fig9_package):
        f(); print("ok", f.__name__)
