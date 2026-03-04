#!/usr/bin/env python3
"""
Benchmark V5 initialization strategies with learning curves and multi-seed runs.

Captures per-epoch metrics (train_loss, val_loss, val_ppl) for each strategy x seed
combination, writes a structured report (.log) and machine-readable data (.json),
and flags grokking candidates automatically.

Usage:
    python scripts/bench_init_strategies.py --samples 2000 --epochs 5
    python scripts/bench_init_strategies.py --strategies golden_ratio,pi,random --num_seeds 5
    python scripts/bench_init_strategies.py --size small --samples 5000 --epochs 10 --num_seeds 3
"""

import json
import math
import sys
import time
import argparse
import statistics
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from v5.model import create_model
from v5.config import get_config
from v5.init import list_strategies
from v5.train import load_tinystories, Trainer


def run_single(
    strategy: str,
    seed: int,
    config: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: Any,
    epochs: int,
) -> Dict[str, Any]:
    """Run one strategy+seed. Returns per-epoch metrics or error."""
    try:
        cfg = deepcopy(config)
        cfg.init_strategy = strategy
        cfg.init_seed = seed
        cfg.max_epochs = epochs

        model = create_model(cfg)
        trainer = Trainer(
            model,
            cfg,
            train_loader,
            val_loader,
            tokenizer=tokenizer,
            checkpoint_dir=str(PROJECT_ROOT / "bench_checkpoints_init"),
            save_checkpoints=False,
            verbose=False,
        )

        seq_len = cfg.max_seq_len
        batch_size = cfg.batch_size
        total_tokens = 0
        start = time.time()
        epoch_metrics = []

        for epoch in range(epochs):
            train_m = trainer.train_epoch(epoch)
            val_m = trainer.validate() if val_loader is not None else {}
            total_tokens += len(train_loader) * batch_size * seq_len
            epoch_metrics.append({
                "epoch": epoch + 1,
                "train_loss": round(train_m["ce_loss"], 4),
                "val_loss": round(val_m.get("val_loss", float("nan")), 4),
                "val_ppl": round(val_m.get("val_ppl", float("nan")), 2),
            })

        elapsed = time.time() - start
        tok_s = int(total_tokens / elapsed) if elapsed > 0 else 0

        return {
            "strategy": strategy,
            "seed": seed,
            "epochs": epoch_metrics,
            "time_s": round(elapsed, 1),
            "tok_s": tok_s,
            "error": None,
        }
    except Exception as e:
        return {
            "strategy": strategy,
            "seed": seed,
            "epochs": [],
            "time_s": 0.0,
            "tok_s": 0,
            "error": str(e),
        }


def detect_grokking(runs: List[Dict]) -> List[str]:
    """Flag runs where val PPL rises then drops below its starting point."""
    candidates = []
    for r in runs:
        if r["error"] or len(r["epochs"]) < 3:
            continue
        ppls = [e["val_ppl"] for e in r["epochs"]]
        if any(math.isnan(p) for p in ppls):
            continue
        start_ppl = ppls[0]
        peak_ppl = start_ppl
        peak_ep = 1
        for i, p in enumerate(ppls[1:], 2):
            if p > peak_ppl:
                peak_ppl = p
                peak_ep = i
        if peak_ppl > start_ppl * 1.05 and peak_ep < len(ppls):
            final_ppl = ppls[-1]
            if final_ppl < start_ppl:
                candidates.append(
                    f"  {r['strategy']} seed={r['seed']}: "
                    f"PPL rose {start_ppl:.1f}->{peak_ppl:.1f} (ep1-{peak_ep}), "
                    f"then dropped to {final_ppl:.1f} (ep{len(ppls)})"
                )
    return candidates


def format_curves(strategy: str, runs: List[Dict], num_epochs: int) -> List[str]:
    """Format per-epoch learning curves for one strategy across seeds."""
    lines = []
    seeds = [r["seed"] for r in runs if not r["error"]]
    if not seeds:
        lines.append(f"--- {strategy}: ALL FAILED ---")
        for r in runs:
            lines.append(f"  seed={r['seed']}: {r['error']}")
        return lines

    lines.append(f"--- {strategy} ({len(seeds)} seed{'s' if len(seeds)>1 else ''}) ---")

    # Header
    hdr_parts = ["Epoch"]
    for s in seeds:
        hdr_parts.append(f"Seed {s} TrL")
        hdr_parts.append(f"ValPPL")
    hdr_parts.append("Mean ValPPL +/- Std")
    lines.append("  " + "  ".join(f"{h:>12}" for h in hdr_parts))

    valid_runs = [r for r in runs if not r["error"]]

    for ep_idx in range(num_epochs):
        parts = [f"{ep_idx+1:>5}"]
        ppls = []
        for r in valid_runs:
            if ep_idx < len(r["epochs"]):
                em = r["epochs"][ep_idx]
                parts.append(f"{em['train_loss']:>12.4f}")
                parts.append(f"{em['val_ppl']:>12.2f}")
                if not math.isnan(em["val_ppl"]):
                    ppls.append(em["val_ppl"])
            else:
                parts.append(f"{'--':>12}")
                parts.append(f"{'--':>12}")
        if len(ppls) >= 2:
            mean_p = statistics.mean(ppls)
            std_p = statistics.stdev(ppls)
            parts.append(f"{mean_p:>10.2f} +/- {std_p:.2f}")
        elif len(ppls) == 1:
            parts.append(f"{ppls[0]:>10.2f}")
        else:
            parts.append(f"{'--':>10}")
        lines.append("  " + "  ".join(parts))

    return lines


def format_summary(all_results: Dict[str, List[Dict]]) -> List[str]:
    """Summary table ranked by mean final val PPL."""
    lines = []
    summary_rows = []

    for strategy, runs in all_results.items():
        valid = [r for r in runs if not r["error"] and r["epochs"]]
        if not valid:
            summary_rows.append({
                "strategy": strategy, "mean_ppl": float("inf"),
                "std_ppl": 0, "best": "FAIL", "worst": "FAIL",
                "avg_time": 0, "n": 0,
            })
            continue
        final_ppls = []
        best_seed, worst_seed = None, None
        best_ppl, worst_ppl = float("inf"), float("-inf")
        for r in valid:
            fp = r["epochs"][-1]["val_ppl"]
            if math.isnan(fp):
                continue
            final_ppls.append(fp)
            if fp < best_ppl:
                best_ppl, best_seed = fp, r["seed"]
            if fp > worst_ppl:
                worst_ppl, worst_seed = fp, r["seed"]

        if not final_ppls:
            continue

        mean_p = statistics.mean(final_ppls)
        std_p = statistics.stdev(final_ppls) if len(final_ppls) >= 2 else 0.0
        avg_time = statistics.mean([r["time_s"] for r in valid])

        summary_rows.append({
            "strategy": strategy, "mean_ppl": mean_p, "std_ppl": std_p,
            "best": f"{best_seed} ({best_ppl:.1f})",
            "worst": f"{worst_seed} ({worst_ppl:.1f})",
            "avg_time": avg_time, "n": len(final_ppls),
        })

    summary_rows.sort(key=lambda x: x["mean_ppl"])

    col = "{:<22} | {:>10} | {:>6} | {:>14} | {:>14} | {:>9}"
    lines.append(col.format("Strategy", "Mean ValPPL", "Std", "Best Seed", "Worst Seed", "Avg Time"))
    lines.append("-" * 90)
    for r in summary_rows:
        if r["n"] == 0:
            lines.append(col.format(r["strategy"], "FAIL", "-", "-", "-", "-"))
        else:
            lines.append(col.format(
                r["strategy"],
                f"{r['mean_ppl']:.2f}",
                f"{r['std_ppl']:.2f}" if r["n"] >= 2 else "-",
                r["best"], r["worst"],
                f"{r['avg_time']:.1f}s",
            ))
    lines.append("-" * 90)

    return lines


def main():
    parser = argparse.ArgumentParser(description="Benchmark V5 init strategies (learning curves)")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--size", type=str, default="tiny",
                        choices=["tiny", "small", "small-matched", "medium", "large"])
    parser.add_argument("--strategies", type=str, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = PROJECT_ROOT / "logs" / f"init_bench_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = log_path.with_suffix(".json")

    strategies = (list_strategies() if args.strategies.lower() == "all"
                  else [s.strip() for s in args.strategies.split(",") if s.strip()])
    if not strategies:
        print("No strategies.", file=sys.stderr)
        sys.exit(1)

    seeds = [args.seed + i for i in range(args.num_seeds)]
    total_runs = len(strategies) * len(seeds)

    print(f"Loading TinyStories (samples={args.samples})...", flush=True)
    train_ds, val_ds, tokenizer = load_tinystories(args.samples, args.seq_len)

    config = get_config(args.size)
    config.max_epochs = args.epochs
    config.batch_size = args.batch_size
    config.max_seq_len = args.seq_len
    config.vocab_size = tokenizer.vocab_size
    model_config = config.to_dict()

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2 if use_cuda else 0, pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2 if use_cuda else 0, pin_memory=use_cuda,
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    bench_config = {
        "samples": args.samples, "batch_size": args.batch_size,
        "seq_len": args.seq_len, "epochs": args.epochs,
        "size": args.size, "seeds": seeds,
        "strategies": strategies, "device": device,
    }

    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Bench: {len(strategies)} strategies x {len(seeds)} seeds = {total_runs} runs, "
          f"{args.epochs} epochs each", flush=True)

    all_results: Dict[str, List[Dict]] = {s: [] for s in strategies}
    all_raw = []
    run_idx = 0
    total_start = time.time()

    for strategy in strategies:
        for seed in seeds:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {strategy} seed={seed}...", end="", flush=True)
            r = run_single(strategy, seed, config, train_loader, val_loader, tokenizer, args.epochs)
            all_results[strategy].append(r)
            all_raw.append(r)
            if r["error"]:
                print(f" FAIL: {r['error']}", flush=True)
            else:
                final = r["epochs"][-1]
                print(f" val_ppl={final['val_ppl']:.2f} ({r['time_s']}s)", flush=True)

    total_elapsed = time.time() - total_start

    # Build report
    lines = []
    lines.append("=" * 90)
    lines.append("V5 Init Strategy Benchmark (Learning Curves)")
    lines.append("=" * 90)
    lines.append(f"Started: {started}")
    lines.append(f"Device: {device}")
    lines.append("")
    lines.append("--- Benchmark config ---")
    for k, v in bench_config.items():
        lines.append(f"  {k} = {v}")
    lines.append("")
    lines.append("--- Model config ---")
    lines.append(json.dumps(model_config, indent=2))
    lines.append("=" * 90)
    lines.append("")

    # Section A: Per-strategy learning curves
    lines.append("SECTION A: PER-STRATEGY LEARNING CURVES")
    lines.append("")
    for strategy in strategies:
        curve_lines = format_curves(strategy, all_results[strategy], args.epochs)
        lines.extend(curve_lines)
        lines.append("")

    # Section B: Summary table
    lines.append("=" * 90)
    lines.append("SECTION B: SUMMARY (ranked by mean final val PPL)")
    lines.append("")
    lines.extend(format_summary(all_results))
    lines.append(f"\nTotal benchmark time: {total_elapsed:.1f}s")
    lines.append("")

    # Section C: Grokking detection
    grok = detect_grokking(all_raw)
    lines.append("=" * 90)
    lines.append("SECTION C: GROKKING CANDIDATES")
    lines.append("")
    if grok:
        lines.extend(grok)
    else:
        lines.append("  None detected (val PPL did not rise then fall below start in any run)")
    lines.append("")
    lines.append("=" * 90)

    report = "\n".join(lines)

    with open(log_path, "w") as f:
        f.write(report)
    print(f"\nReport: {log_path}", flush=True)

    # JSON output
    json_data = {
        "bench_config": bench_config,
        "model_config": model_config,
        "started": started,
        "total_time_s": round(total_elapsed, 1),
        "results": all_raw,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON:   {json_path}", flush=True)

    # Print summary to stdout
    print("\n" + "\n".join(format_summary(all_results)), flush=True)


if __name__ == "__main__":
    main()
