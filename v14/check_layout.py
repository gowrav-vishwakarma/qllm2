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
    "l1_data.py",        # L1: document builder + frozen-mind reader (data prep)
    "l1_key.py",         # L1: the key head, its training loop and the harness hook (raw torch)
    "train_pool.py",     # L1: tokenised web text for the key head (data prep)
    "l3_data.py",        # L3: cross-document pair builder (data prep)
    "shared_fetch.py",   # SL1: shared-library text fetch eval (numpy / raw torch embedder)
    "a5_text.py",        # A5: synonym-cell text eval (MiniLM CPU thread cap)
    "a6_reread.py",      # A6: warm / cold re-read eval (frozen-mind decode loop, raw torch)
    "a11_keyswap.py",    # A11: key-swap relevance eval (frozen-mind short reads, raw torch)
    "a13_manykey.py",    # A13: many-key fetch eval (frozen states, max-sim, raw torch)
    "package_fetch.py",  # the package fetch as a harness key_fn (two organs, raw torch)
    "pack.py",           # library packs: text-first format + install tool (raw torch / numpy)
    "pack_check.py",     # cross-document check of the install tool
    "n11_refresh.py",    # N11-R: refresh-rule eval (raw torch, M2 full forwards)
    "sr0.py",            # SR-0: rule-vector transplant test (eval harness, raw torch)
    "w0_gate.py",        # W0: eval harness (surprise scoring on frozen checkpoints)
    "ngram.py",          # N7: integer hashing / sort / scatter (declared boundary, like v13 cond_mem.py)
})

KERNEL_FUNCTIONS: dict[str, frozenset[str]] = {   # N1's kernel boundaries left with N1 (9e0a3fd)
    "units.py": frozenset({
        "_echo_read",        # N9a: v13's echo read (raw-torch boundary there too) + content phases
        "_content_phase",    # N9a: integer ids -> angles (masked gather over -1 padding)
        "_turn",             # N9a: channel-pair rotation of raw q / k
        "_circles",          # N10: coordinate -> unit complex on P circles (cos / sin)
        "_salient_phase",    # N9s: importance gates + running salient angle (raw, sempyt scan)
        "_surprise",         # N12: body surprise through the tied head (chunked logsumexp, no grad)
        "_item_read",        # N12: echo's delta memory with surprise-selected writes (raw, as echo)
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
