#!/usr/bin/env python3
"""
Plot training curves from v6.train_transformer_baseline logs.

That trainer prints batch lines without div/wdiv and epoch lines with
"| tok/s | Time: ..." — different from PAM v6.train. Reuses plot_training.plot_log.

Usage:
  python scripts/plot_transformer_baseline.py logs/v6/.../transformer_baseline.log
  python scripts/plot_transformer_baseline.py path/to.log --live --x-axis tokens
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from plot_training import live_png_path, parse_header, plot_log

# [1] 0/19277 (0%) loss=... ppl=... lr=... | 638 tok/s (avg 638) ETA 0m00s | GPU ... | gtok=...
BATCH_RE_TB = re.compile(
    r"\[(\d+)\]\s+(\d+)/(\d+)\s+\((\d+)%\)\s+"
    r"loss=([\d.e+-]+)\s+ppl=([\d.e+-]+)\s+lr=([\d.e+-]+)\s+\|\s+"
    r"([\d.]+)\s+tok/s\s+\(avg\s+([\d.]+)\)\s+ETA\s+[\d]+m[\d]+s"
    r"(?:\s+\|\s+GPU\s+[\d.]+/[\d.]+GB)?"
    r"(?:\s+\|\s+gtok=(\d+))?"
)

# Epoch 1/10 | Train Loss: 4.9256 PPL: 137.77 | 92356 tok/s | Time: 1282.4s (118,437,888 tok) | Val Loss: 3.9842 PPL: 53.74 *best*
EPOCH_RE_TB = re.compile(
    r"Epoch (\d+)/(\d+) \| Train Loss: ([\d.]+) PPL: ([\d.]+) \| "
    r"[\d.]+\s+tok/s \| Time: ([\d.]+)s(?:\s+\([\d,]+\s+tok\))? \| "
    r"Val Loss: ([\d.]+) PPL: ([\d.]+)(?:\s+\*best\*)?"
)

ARCH_RE_TB = re.compile(
    r"Architecture: d_model=(\d+), n_layers=(\d+), n_heads=(\d+), d_ff=(\d+)"
)


def parse_transformer_baseline_log(path: Path) -> dict:
    """Same structure as plot_training.parse_log."""
    text = path.read_text()
    batches = []
    epochs = []
    for line in text.splitlines():
        m = BATCH_RE_TB.search(line)
        if m:
            ep, bidx, total, _pct, loss, ppl, lr, _inst_tok, avg_tok, gtok = m.groups()
            entry = {
                "epoch": int(ep),
                "batch_idx": int(bidx),
                "total_batches": int(total),
                "loss": float(loss),
                "ppl": float(ppl),
                "lr": float(lr),
                "samples_per_s": 0.0,
                "tok_per_s": int(float(avg_tok)),
            }
            if gtok is not None:
                entry["gtok"] = int(gtok)
            batches.append(entry)
            continue
        m = EPOCH_RE_TB.search(line)
        if m:
            ep, max_ep, train_loss, train_ppl, epoch_time, val_loss, val_ppl = m.groups()
            epochs.append(
                {
                    "epoch": int(ep),
                    "max_epochs": int(max_ep),
                    "train_loss": float(train_loss),
                    "train_ppl": float(train_ppl),
                    "epoch_time_s": float(epoch_time),
                    "val_loss": float(val_loss),
                    "val_ppl": float(val_ppl),
                }
            )
    return {"batches": batches, "epochs": epochs}


def enrich_header(path: Path, header: dict) -> dict:
    """Add transformer-baseline-specific subtitle fields."""
    text = path.read_text()
    m = ARCH_RE_TB.search(text)
    if m:
        header = {**header, "baseline_arch": f"d={m.group(1)} L={m.group(2)} h={m.group(3)} ff={m.group(4)}"}
    return header


def _make_info_text(h: dict) -> str:
    parts = []
    if h.get("baseline_arch"):
        parts.append(h["baseline_arch"])
    if h.get("size"):
        parts.append(f"Config: {h['size']}")
    if h.get("dim"):
        parts.append(f"dim={h['dim']}, state={h.get('state_dim', '?')}")
    if h.get("layers"):
        parts.append(f"layers={h['layers']}, banks={h.get('banks', '?')}")
    if h.get("attn_every"):
        parts.append(f"attn_every={h['attn_every']}")
    if h.get("max_samples"):
        parts.append(f"samples={h['max_samples']}")
    if h.get("init_strategy"):
        parts.append(f"init={h['init_strategy']}")
    if h.get("total_params"):
        parts.append(f"params={h['total_params']}")
    if h.get("batches_per_epoch"):
        parts.append(f"batches/epoch={h['batches_per_epoch']}")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot transformer baseline (v6.train_transformer_baseline) training curves from logs"
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Log file path(s)")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PNG path (default: first log's directory, <stem>.png; with --live: <stem>-live.png)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Re-read log and refresh periodically; save PNG each refresh (<stem>-live.png next to first log unless -o)",
    )
    parser.add_argument("--live-interval", type=int, default=10, help="Refresh interval in seconds for --live")
    parser.add_argument(
        "--x-axis",
        type=str,
        default="batch",
        choices=["batch", "tokens"],
        help="X-axis unit: 'batch' (default) or 'tokens' (requires gtok= in logs)",
    )
    args = parser.parse_args()

    all_logs: list[Path] = []
    for p in args.logs:
        if "*" in str(p):
            all_logs.extend(sorted(Path().glob(str(p))))
        elif p.exists():
            all_logs.append(p)
        else:
            print(f"Warning: {p} not found, skipping")
    if not all_logs:
        print("No log files found.")
        return 1

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_logs), 10)))
    header = enrich_header(all_logs[0], parse_header(all_logs[0]))
    bpe = int(header.get("batches_per_epoch", 1))
    max_epochs = int(header.get("epochs", 10))

    def _add_epoch_axis(ax):
        def batch_to_epoch(b):
            return b / bpe + 1 if bpe else 1

        def epoch_to_batch(e):
            return (e - 1) * bpe if bpe else 0

        ax2 = ax.secondary_xaxis("top", functions=(batch_to_epoch, epoch_to_batch))
        ax2.set_xlabel("Epoch")
        ax2.set_ticks(range(1, max_epochs + 1))

    def _add_epoch_boundaries(ax):
        for ep in range(2, max_epochs + 1):
            batch_at_boundary = (ep - 1) * bpe
            ax.axvline(batch_at_boundary, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    def plot_wall_time(ax_time, all_data: list, labels: list, colors_list):
        if not all_data or not any(d["epochs"] for d in all_data):
            return
        max_epochs_seen = max((e["epoch"] for d in all_data for e in d["epochs"]), default=0)
        if max_epochs_seen == 0:
            return
        x = np.arange(1, max_epochs_seen + 1)
        width = 0.8 / len(all_data) if len(all_data) > 1 else 0.7
        for i, (data, label) in enumerate(zip(all_data, labels)):
            epochs = data.get("epochs", [])
            times = [0.0] * max_epochs_seen
            for e in epochs:
                if 1 <= e["epoch"] <= max_epochs_seen:
                    times[e["epoch"] - 1] = e.get("epoch_time_s", 0) / 60
            offset = (i - (len(all_data) - 1) / 2) * width if len(all_data) > 1 else 0
            ax_time.bar(
                x + offset,
                times,
                width,
                label=label,
                color=colors_list[i % len(colors_list)],
                alpha=0.85,
            )
        ax_time.set_xlabel("Epoch")
        ax_time.set_ylabel("Wall time (min)")
        ax_time.set_title("Wall Time per Epoch")
        ax_time.legend(loc="upper right", fontsize=8)
        ax_time.grid(True, alpha=0.3, axis="y")

    use_tokens = args.x_axis == "tokens"
    x_label = "Tokens (M)" if use_tokens else "Batch"

    def do_plot():
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6])
        ax_loss = fig.add_subplot(gs[0, 0])
        ax_ppl = fig.add_subplot(gs[0, 1])
        ax_lr = fig.add_subplot(gs[1, 0])
        ax_tok = fig.add_subplot(gs[1, 1])
        ax_time = fig.add_subplot(gs[2, :])

        all_data = []
        for i, log_path in enumerate(all_logs):
            data = parse_transformer_baseline_log(log_path)
            all_data.append(data)
            label = log_path.stem.replace("v5_train_", "")
            plot_log(data, label, colors[i % len(colors)], ax_loss, ax_ppl, ax_lr, ax_tok, use_tokens=use_tokens)

        plot_wall_time(ax_time, all_data, [p.stem.replace("v5_train_", "") for p in all_logs], colors)

        if not use_tokens:
            for ax in (ax_loss, ax_ppl, ax_lr, ax_tok):
                _add_epoch_boundaries(ax)
                _add_epoch_axis(ax)

        for ax in (ax_loss, ax_ppl, ax_lr, ax_tok):
            ax.set_xlabel(x_label)

        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Train & Val Loss")
        ax_loss.legend(loc="upper right", fontsize=8)
        ax_loss.grid(True, alpha=0.3)

        ax_ppl.set_ylabel("PPL")
        ax_ppl.set_title("Train & Val PPL (log scale)")
        ax_ppl.set_yscale("log")
        ax_ppl.legend(loc="upper right", fontsize=8)
        ax_ppl.grid(True, alpha=0.3)

        ax_lr.set_ylabel("Learning rate")
        ax_lr.set_title("LR Schedule")
        ax_lr.legend(loc="upper right", fontsize=8)
        ax_lr.grid(True, alpha=0.3)

        ax_tok.set_ylabel("tok/s")
        ax_tok.set_title("Throughput")
        ax_tok.legend(loc="lower right", fontsize=8)
        ax_tok.grid(True, alpha=0.3)

        info_text = _make_info_text(header)
        title_prefix = "Transformer baseline (WikiText-103)"
        if info_text:
            fig.suptitle(f"{title_prefix} | {info_text}", fontsize=9, y=1.02)
        else:
            fig.suptitle(title_prefix, fontsize=9, y=1.02)

        plt.tight_layout()
        return fig

    if args.live:
        live_out = live_png_path(all_logs[0], args.output)
        fig = plt.figure(figsize=(12, 10))

        def update(_frame):
            nonlocal header, bpe, max_epochs
            header = enrich_header(all_logs[0], parse_header(all_logs[0]))
            bpe = int(header.get("batches_per_epoch", 1))
            max_epochs = int(header.get("epochs", 10))
            fig.clear()
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6])
            ax_loss = fig.add_subplot(gs[0, 0])
            ax_ppl = fig.add_subplot(gs[0, 1])
            ax_lr = fig.add_subplot(gs[1, 0])
            ax_tok = fig.add_subplot(gs[1, 1])
            ax_time = fig.add_subplot(gs[2, :])
            all_data = []
            for i, log_path in enumerate(all_logs):
                data = parse_transformer_baseline_log(log_path)
                all_data.append(data)
                label = log_path.stem.replace("v5_train_", "")
                plot_log(data, label, colors[i % len(colors)], ax_loss, ax_ppl, ax_lr, ax_tok, use_tokens=use_tokens)
            plot_wall_time(ax_time, all_data, [p.stem.replace("v5_train_", "") for p in all_logs], colors)
            if not use_tokens:
                for ax in (ax_loss, ax_ppl, ax_lr, ax_tok):
                    for ep in range(2, max_epochs + 1):
                        ax.axvline((ep - 1) * bpe, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
                    ax2 = ax.secondary_xaxis(
                        "top",
                        functions=(
                            lambda b: b / bpe + 1 if bpe else 1,
                            lambda e: (e - 1) * bpe if bpe else 0,
                        ),
                    )
                    ax2.set_xlabel("Epoch")
                    ax2.set_ticks(range(1, max_epochs + 1))
            for ax in (ax_loss, ax_ppl, ax_lr, ax_tok):
                ax.set_xlabel(x_label)
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("Train & Val Loss")
            ax_loss.legend(loc="upper right", fontsize=8)
            ax_loss.grid(True, alpha=0.3)
            ax_ppl.set_ylabel("PPL")
            ax_ppl.set_title("Train & Val PPL (log scale)")
            ax_ppl.set_yscale("log")
            ax_ppl.legend(loc="upper right", fontsize=8)
            ax_ppl.grid(True, alpha=0.3)
            ax_lr.set_ylabel("Learning rate")
            ax_lr.set_title("LR Schedule")
            ax_lr.legend(loc="upper right", fontsize=8)
            ax_lr.grid(True, alpha=0.3)
            ax_tok.set_ylabel("tok/s")
            ax_tok.set_title("Throughput")
            ax_tok.legend(loc="lower right", fontsize=8)
            ax_tok.grid(True, alpha=0.3)
            info_text = _make_info_text(header)
            title_prefix = "Transformer baseline (WikiText-103)"
            if info_text:
                fig.suptitle(f"{title_prefix} | {info_text}", fontsize=9, y=1.02)
            else:
                fig.suptitle(title_prefix, fontsize=9, y=1.02)
            plt.tight_layout()
            fig.savefig(live_out, dpi=120)

        print(
            f"Live mode: refreshing every {args.live_interval}s; saving to {live_out} each refresh. "
            "Close the window to exit."
        )
        update(0)
        ani = animation.FuncAnimation(
            fig,
            update,
            interval=args.live_interval * 1000,
            cache_frame_data=False,
        )
        plt.show()
        return 0

    fig = do_plot()
    out = args.output
    if out is None:
        log0 = all_logs[0]
        out = log0.parent / f"{log0.stem}.png"
    fig.savefig(out, dpi=120)
    print(f"Saved: {out}")
    if args.show:
        plt.show()
    else:
        plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
