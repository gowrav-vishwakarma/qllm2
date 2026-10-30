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
    "ngram.py",          # N7: integer hashing / sort / scatter (declared boundary, like v13 cond_mem.py)
})

KERNEL_FUNCTIONS: dict[str, frozenset[str]] = {   # N1's kernel boundaries left with N1 (9e0a3fd)
    "units.py": frozenset({
        "_echo_read",        # N9a: v13's echo read (raw-torch boundary there too) + content phases
        "_content_phase",    # N9a: integer ids -> angles (masked gather over -1 padding)
        "_turn",             # N9a: channel-pair rotation of raw q / k
        "_circles",          # N10: coordinate -> unit complex on P circles (cos / sin)
        "__init__",          # init pairing: fork_rng so the body and RNG stay as plain echo
    }),
}


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
