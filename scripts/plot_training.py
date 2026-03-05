#!/usr/bin/env python3
"""
Plot training curves (loss, PPL, LR, throughput) from V5 training log files.

Parses batch-level and epoch-level lines without modifying the training code.
Usage:
  python scripts/plot_training.py logs/v5_train_small-matched.log
  python scripts/plot_training.py logs/v5_train_*.log --show
"""

import argparse
import re
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# Batch: [1] batch 50/4806 loss=9.4727 ppl=13000.0 div=0.0000 lr=1.00e-04 | 57.6 samples/s | 14736 tok/s
BATCH_RE = re.compile(
    r"\[(\d+)\] batch (\d+)/(\d+) loss=([\d.]+) ppl=([\d.]+) "
    r"div=[\d.]+ lr=([\d.e+-]+) \| ([\d.]+) samples/s \| (\d+) tok/s"
)

# Epoch: Epoch 1/10 | Train Loss: 3.7232 PPL: 41.40 | Time: 1220.0s | Val Loss: 2.9379 PPL: 18.88 *best*
EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+) \| Train Loss: ([\d.]+) PPL: ([\d.]+) \| "
    r"Time: ([\d.]+)s \| Val Loss: ([\d.]+) PPL: ([\d.]+)"
)

# Header lines for architecture info
HEADER_RE = {
    "size": re.compile(r"Size: (\S+)"),
    "dim": re.compile(r"Complex dim: (\d+)"),
    "state_dim": re.compile(r"SSM state dim: (\d+)"),
    "layers": re.compile(r"Layers: (\d+)"),
    "banks": re.compile(r"Banks: (\d+)"),
    "attn_every": re.compile(r"Attention every: (\d+)"),
    "epochs": re.compile(r"Epochs: (\d+)"),
    "max_samples": re.compile(r"Max samples: (\d+)"),
    "init_strategy": re.compile(r"Init strategy: (.+)"),
    "total_params": re.compile(r"Total: ([\d,]+)"),
    "batches_per_epoch": re.compile(r"Batches/epoch: (\d+)"),
}


def parse_header(path: Path) -> dict:
    """Parse log header for architecture and run info."""
    text = path.read_text()
    info = {}
    for key, pat in HEADER_RE.items():
        m = pat.search(text)
        if m:
            info[key] = m.group(1)
    return info


def parse_log(path: Path) -> dict:
    """Parse a log file and return batch-level and epoch-level data."""
    text = path.read_text()
    batches = []
    epochs = []
    for line in text.splitlines():
        m = BATCH_RE.search(line)
        if m:
            ep, bidx, total, loss, ppl, lr, sps, toks = m.groups()
            batches.append({
                "epoch": int(ep),
                "batch_idx": int(bidx),
                "total_batches": int(total),
                "loss": float(loss),
                "ppl": float(ppl),
                "lr": float(lr),
                "samples_per_s": float(sps),
                "tok_per_s": int(toks),
            })
            continue
        m = EPOCH_RE.search(line)
        if m:
            ep, max_ep, train_loss, train_ppl, epoch_time, val_loss, val_ppl = m.groups()
            epochs.append({
                "epoch": int(ep),
                "max_epochs": int(max_ep),
                "train_loss": float(train_loss),
                "train_ppl": float(train_ppl),
                "epoch_time_s": float(epoch_time),
                "val_loss": float(val_loss),
                "val_ppl": float(val_ppl),
            })
    return {"batches": batches, "epochs": epochs}


def global_batch_idx(b: dict, batches_per_epoch: int) -> int:
    """Convert (epoch, batch_idx) to global batch index."""
    return (b["epoch"] - 1) * batches_per_epoch + b["batch_idx"]


def smooth(x: np.ndarray, window: int = 51) -> np.ndarray:
    """Moving average smoothing. window must be odd."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_log(data: dict, label: str, color: str, ax_loss, ax_ppl, ax_lr, ax_tok):
    """Add one run's curves to the four axes."""
    batches = data["batches"]
    epochs = data["epochs"]
    if not batches:
        return
    bpe = batches[0]["total_batches"]
    global_indices = [global_batch_idx(b, bpe) for b in batches]
    loss_vals = np.array([b["loss"] for b in batches])
    ppl_vals = np.array([b["ppl"] for b in batches])
    lr_vals = np.array([b["lr"] for b in batches])
    tok_vals = np.array([b["tok_per_s"] for b in batches])

    # Smooth (reduce points to match)
    w = min(51, len(loss_vals) // 2 | 1)
    if w % 2 == 0:
        w -= 1
    if w >= 3:
        pad = (w - 1) // 2
        loss_smooth = smooth(loss_vals, w)
        ppl_smooth = smooth(ppl_vals, w)
        idx_smooth = global_indices[pad : len(global_indices) - pad]
        ax_loss.plot(idx_smooth, loss_smooth, color=color, label=f"{label} (train)", alpha=0.9)
        ax_ppl.plot(idx_smooth, ppl_smooth, color=color, label=f"{label} (train)", alpha=0.9)
    else:
        ax_loss.plot(global_indices, loss_vals, color=color, label=f"{label} (train)", alpha=0.9)
        ax_ppl.plot(global_indices, ppl_vals, color=color, label=f"{label} (train)", alpha=0.9)

    ax_lr.plot(global_indices, lr_vals, color=color, label=label, alpha=0.8)
    ax_tok.plot(global_indices, tok_vals, color=color, label=label, alpha=0.8)

    # Epoch-level validation (dots)
    if epochs:
        ep_indices = [(e["epoch"] - 0.5) * bpe for e in epochs]
        val_loss = [e["val_loss"] for e in epochs]
        val_ppl = [e["val_ppl"] for e in epochs]
        ax_loss.scatter(ep_indices, val_loss, color=color, s=40, marker="o", zorder=5, label=f"{label} (val)")
        ax_ppl.scatter(ep_indices, val_ppl, color=color, s=40, marker="o", zorder=5, label=f"{label} (val)")


def main():
    parser = argparse.ArgumentParser(description="Plot V5 training curves from log files")
    parser.add_argument("logs", nargs="+", type=Path, help="Log file path(s)")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output PNG path (default: next to first log)")
    parser.add_argument("--live", action="store_true", help="Re-read log and refresh every 10s (for live training)")
    parser.add_argument("--live-interval", type=int, default=10, help="Refresh interval in seconds for --live")
    args = parser.parse_args()

    # Expand globs
    all_logs = []
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
    header = parse_header(all_logs[0])
    bpe = int(header.get("batches_per_epoch", 1))
    max_epochs = int(header.get("epochs", 10))

    def _add_epoch_axis(ax):
        """Add secondary x-axis showing epoch on top."""
        def batch_to_epoch(b):
            return b / bpe + 1 if bpe else 1

        def epoch_to_batch(e):
            return (e - 1) * bpe if bpe else 0

        ax2 = ax.secondary_xaxis("top", functions=(batch_to_epoch, epoch_to_batch))
        ax2.set_xlabel("Epoch")
        ax2.set_ticks(range(1, max_epochs + 1))

    def _add_epoch_boundaries(ax):
        """Add vertical dashed lines at epoch boundaries."""
        for ep in range(2, max_epochs + 1):
            batch_at_boundary = (ep - 1) * bpe
            ax.axvline(batch_at_boundary, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    def _make_info_text(h: dict) -> str:
        parts = []
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
        return " | ".join(parts)

    def plot_wall_time(ax_time, all_data: list, labels: list, colors_list):
        """Bar chart of wall time per epoch for each run."""
        if not all_data or not any(d["epochs"] for d in all_data):
            return
        max_epochs_seen = max(
            (e["epoch"] for d in all_data for e in d["epochs"]),
            default=0,
        )
        if max_epochs_seen == 0:
            return
        x = np.arange(1, max_epochs_seen + 1)
        width = 0.8 / len(all_data) if len(all_data) > 1 else 0.7
        for i, (data, label) in enumerate(zip(all_data, labels)):
            epochs = data.get("epochs", [])
            times = [0.0] * max_epochs_seen
            for e in epochs:
                if 1 <= e["epoch"] <= max_epochs_seen:
                    times[e["epoch"] - 1] = e.get("epoch_time_s", 0) / 60  # minutes
            offset = (i - (len(all_data) - 1) / 2) * width if len(all_data) > 1 else 0
            ax_time.bar(x + offset, times, width, label=label, color=colors_list[i % len(colors_list)], alpha=0.85)
        ax_time.set_xlabel("Epoch")
        ax_time.set_ylabel("Wall time (min)")
        ax_time.set_title("Wall Time per Epoch")
        ax_time.legend(loc="upper right", fontsize=8)
        ax_time.grid(True, alpha=0.3, axis="y")

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
            data = parse_log(log_path)
            all_data.append(data)
            label = log_path.stem.replace("v5_train_", "")
            plot_log(data, label, colors[i % len(colors)], ax_loss, ax_ppl, ax_lr, ax_tok)

        plot_wall_time(ax_time, all_data, [p.stem.replace("v5_train_", "") for p in all_logs], colors)

        for ax in (ax_loss, ax_ppl, ax_lr, ax_tok):
            _add_epoch_boundaries(ax)
            _add_epoch_axis(ax)
            ax.set_xlabel("Batch")

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
        if info_text:
            fig.suptitle(f"V5 Algebraic LM | {info_text}", fontsize=9, y=1.02)

        plt.tight_layout()
        return fig

    if args.live:
        fig = plt.figure(figsize=(12, 10))

        def update(_frame):
            nonlocal header, bpe, max_epochs
            header = parse_header(all_logs[0])
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
                data = parse_log(log_path)
                all_data.append(data)
                label = log_path.stem.replace("v5_train_", "")
                plot_log(data, label, colors[i % len(colors)], ax_loss, ax_ppl, ax_lr, ax_tok)
            plot_wall_time(ax_time, all_data, [p.stem.replace("v5_train_", "") for p in all_logs], colors)
            for ax in (ax_loss, ax_ppl, ax_lr, ax_tok):
                for ep in range(2, max_epochs + 1):
                    ax.axvline((ep - 1) * bpe, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
                ax2 = ax.secondary_xaxis("top", functions=(
                    lambda b: b / bpe + 1 if bpe else 1,
                    lambda e: (e - 1) * bpe if bpe else 0,
                ))
                ax2.set_xlabel("Epoch")
                ax2.set_ticks(range(1, max_epochs + 1))
                ax.set_xlabel("Batch")
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
            if info_text:
                fig.suptitle(f"V5 Algebraic LM | {info_text}", fontsize=9, y=1.02)
            plt.tight_layout()

        print(f"Live mode: refreshing every {args.live_interval}s. Close the window to exit.")
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
        out = all_logs[0].with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"Saved: {out}")
    if args.show:
        plt.show()
    else:
        plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
