"""Fail if v14 reaches around sempyt instead of through it.

Same rules as ``v13_sempty/check_torch_layout.py`` (it does the scanning);
only the root and the declared raw-torch boundaries are v14's own.

    .venv/bin/python -m v14.check_layout
"""

from __future__ import annotations

import sys
from pathlib import Path

from v13_sempty import check_torch_layout as guard

ROOT = Path(__file__).resolve().parent

SKIP_FILES = frozenset({
    "selftest.py",       # compares against raw torch references
    "microbench.py",     # data generation + training loop
    "check_layout.py",
})

KERNEL_FUNCTIONS: dict[str, frozenset[str]] = {}   # N1's kernel boundaries left with N1 (9e0a3fd)


def main() -> int:
    guard.ROOT, guard.SKIP_FILES, guard.KERNEL_FUNCTIONS = ROOT, SKIP_FILES, KERNEL_FUNCTIONS
    hits = [h for path in sorted(ROOT.glob("**/*.py"))
            if "tmp" not in path.parts for h in guard.scan_file(path)]
    if hits:
        print("v14 is reaching around sempyt:\n")
        print("\n".join(f"  {hit}" for hit in hits))
        print(f"\n{len(hits)} violation(s).")
        return 1
    print("OK — v14 is named end to end outside declared boundaries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
