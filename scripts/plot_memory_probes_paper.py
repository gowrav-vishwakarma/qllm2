#!/usr/bin/env python3
"""Generate paper figures exclusively from publication result JSON."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get('schema_version') != 'memory-probes-publication/v1':
        raise ValueError(f'{path} is not a publication-v1 result')
    return payload


def _group(rows: list[dict], prefix_len: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row['key'][:prefix_len])
        grouped[key].append(row)
    return grouped


def _plot_curve(ax, rows: list[dict], label: str) -> None:
    rows = sorted(rows, key=lambda row: row['key'][-1])
    x = [row['key'][-1] for row in rows]
    y = [row['mean'] for row in rows]
    ci = [row['ci95'] for row in rows]
    ax.plot(x, y, marker='o', markersize=3, label=label)
    ax.fill_between(x, [a - b for a, b in zip(y, ci)],
                    [a + b for a, b in zip(y, ci)], alpha=0.18)


def plot_capacity(payload: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)

    equal = _group(payload['aggregates']['binding_equal_width'], 2)
    for (d, method), rows in equal.items():
        _plot_curve(axes[0], rows, f'{method}, d={d}')
    axes[0].set_title('Equal embedding width')
    axes[0].set_xlabel('Associations written')
    axes[0].set_ylabel('Top-1 retrieval accuracy')
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=7)

    matched = _group(payload['aggregates']['binding_matched_bytes'], 3)
    for (state_bytes, matrix_d, method), rows in matched.items():
        _plot_curve(
            axes[1], rows,
            f'{method}, {state_bytes / 1024:.1f} KiB (PAM d={matrix_d})',
        )
    axes[1].set_title('Matched state bytes')
    axes[1].set_xlabel('Associations written')
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(fontsize=7)

    fig.savefig(output_dir / 'capacity.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'capacity.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_persistence(payload: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    groups = _group(payload['aggregates']['persistence'], 2)
    for (d, gamma), rows in groups.items():
        _plot_curve(ax, rows, f'd={d}, gamma={gamma}')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Filler distance')
    ax.set_ylabel('Relative target alignment')
    ax.set_title('Persistence under decay and interference')
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(output_dir / 'persistence.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'persistence.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_rank(payload: dict, output_dir: Path) -> None:
    rows = [
        row for row in payload['aggregates']['rank']
        if row['key'][-1] == 'final'
    ]
    labels = [f'd={r["key"][0]} {r["key"][1]}' for r in rows]
    means = [r['mean'] for r in rows]
    cis = [r['ci95'] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.bar(range(len(rows)), means, yerr=cis, capsize=3)
    ax.set_xticks(range(len(rows)), labels, rotation=30, ha='right')
    ax.set_ylabel('Final effective rank')
    ax.set_title('State utilization under synthetic writes')
    fig.savefig(output_dir / 'rank.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'rank.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('memory_probes/paper/figures'),
    )
    args = parser.parse_args()
    payload = _load(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_capacity(payload, args.output_dir)
    plot_persistence(payload, args.output_dir)
    plot_rank(payload, args.output_dir)
    print(f'Figures saved to {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
