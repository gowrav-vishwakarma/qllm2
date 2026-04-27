#!/usr/bin/env python3
"""
Extract validation PPL at epochs 1, 2, and the last completed epoch from training logs
under logs/v6 .. logs/v9. Supports standard V6-V9 epoch summary lines and v6
scaling_sweep tabular logs (epoch, step, phase, val_ppl).

Usage:
  uv run python scripts/extract_val_ppl_epochs.py           # TSV to stdout
  uv run python scripts/extract_val_ppl_epochs.py --json    # JSON lines
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# One line: Epoch k/n ... Val Loss: ... PPL: x.xx  OR  Validation/Test Loss
EPOCH_LINE = re.compile(
    r"Epoch (\d+)/(\d+).*?Val(?:idation)?(?:/Test)?\s+Loss:\s+[\d.eE+-]+\s+PPL:\s+([\d.]+)"
)


def parse_standard_log(text: str) -> dict[int, float] | None:
    """Map epoch index -> val PPL (last occurrence per epoch wins)."""
    by_ep: dict[int, float] = {}
    for m in EPOCH_LINE.finditer(text):
        ep, _total, ppl = m.group(1), m.group(2), float(m.group(3))
        by_ep[int(ep)] = ppl
    return by_ep if by_ep else None


def parse_scaling_sweep(text: str) -> dict[int, float] | None:
    """
    Tab-separated: epoch, step, mid|end, val_ppl
    Prefer 'end' row per epoch; else last row of that epoch.
    """
    by_ep_end: dict[int, float] = {}
    by_ep_last: dict[int, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            ep = int(parts[0])
            phase = parts[2]
            ppl = float(parts[3])
        except (ValueError, IndexError):
            continue
        by_ep_last[ep] = ppl
        if phase == "end":
            by_ep_end[ep] = ppl
    if not by_ep_last:
        return None
    out: dict[int, float] = {}
    for ep in by_ep_last:
        out[ep] = by_ep_end.get(ep, by_ep_last[ep])
    return out


def triple_from_epochs(by_ep: dict[int, float]) -> tuple[str, str, str, int | None, str | None]:
    """e1, e2, e_last, last_epoch, error."""
    if not by_ep:
        return "—", "—", "—", None, "no_epoch_parsed"
    max_ep = max(by_ep)
    e1 = f"{by_ep[1]:.2f}" if 1 in by_ep else "—"
    e2 = f"{by_ep[2]:.2f}" if 2 in by_ep else "—"
    el = f"{by_ep[max_ep]:.2f}"
    return e1, e2, el, max_ep, None


def process_file(path: Path, repo: Path) -> dict:
    rel = path.relative_to(repo)
    text = path.read_text(encoding="utf-8", errors="replace")
    by_ep = parse_standard_log(text)
    kind = "standard"
    if by_ep is None:
        by_ep2 = parse_scaling_sweep(text)
        if by_ep2:
            by_ep = by_ep2
            kind = "scaling_sweep"
    e1, e2, el, last_ep, err = triple_from_epochs(by_ep or {})
    wikitext = "wikitext103" in str(rel).lower() or "wikitext" in str(rel).lower()
    if "tinystories" in str(rel).lower():
        wikitext = False
    note: list[str] = []
    if kind == "scaling_sweep":
        note.append("scaling_sweep log format")
    if by_ep and (lo := min(by_ep)) > 1:
        miss = f"ep1–ep{lo - 1}" if lo > 2 else "ep1"
        note.append(f"log fragment: first epoch in file is {lo} ({miss} missing)")
    if not wikitext and "tinystories" not in str(rel).lower():
        if "tiny" in str(rel).lower() or "smoke" in str(rel).lower() or "mac_cpu" in str(rel).lower():
            note.append("non-WT103 / tiny or smoke")
    elif not wikitext and "tinystories" in str(rel).lower():
        note.append("TinyStories")
    if err:
        note.append(err)
    return {
        "path": str(rel).replace("\\", "/"),
        "e1": e1,
        "e2": e2,
        "e_last": el,
        "last_epoch": last_ep,
        "kind": kind,
        "note": "; ".join(note) if note else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print JSONL")
    ap.add_argument("--root", type=Path, default=REPO, help="repo root")
    args = ap.parse_args()
    repo = args.root.resolve()
    files: list[Path] = []
    for root in [repo / "logs" / f"v{i}" for i in (6, 7, 8, 9)]:
        if not root.is_dir():
            continue
        files.extend(sorted(root.rglob("*.log")))

    rows = [process_file(p, repo) for p in files]
    rows.sort(key=lambda r: (r["path"]))

    if args.json:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        return

    # TSV
    print("path\te1\te2\te_last\tlast_ep\tkind\tnote")
    for r in rows:
        le = r["last_epoch"]
        le_s = str(le) if le is not None else ""
        print(
            f"{r['path']}\t{r['e1']}\t{r['e2']}\t{r['e_last']}\t{le_s}\t{r['kind']}\t{r['note']}"
        )


if __name__ == "__main__":
    main()
