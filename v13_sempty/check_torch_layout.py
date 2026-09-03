"""Fail if v13_sempty reaches around sempyt instead of through it.

Two classes of violation:

  layout   ``.transpose`` / ``.unsqueeze`` / ``.permute`` / ``.view`` /
           ``.reshape`` / ``.contiguous`` / ``[..., 0]`` — positional axis work
           that has a named equivalent (``.to`` / ``.alias`` / ``contract`` /
           ``real`` / ``imag`` / ``as_complex``).

  escape   ``.data`` / ``.raw(`` / ``torch.<something>`` — leaving NamedTensor
           for raw torch. Legal only at a declared boundary.

Boundaries are declared per file in ``KERNEL_FUNCTIONS`` (a named function that
wraps a kernel sempyt cannot express) or in ``SKIP_FILES`` (a whole module that
is a boundary: autograd Functions, the v13 comparison harness, training entry
points). Everything else must be named end to end.

Run until clean:
    .venv/bin/python -m v13_sempty.check_torch_layout
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Whole modules that are boundaries by design.
SKIP_FILES = frozenset({
    "fused_ce.py",        # custom autograd Function: chunked [N, vocab] on purpose
    "triton_kernels.py",  # fused real-PAM scan: Triton kernels + autograd Function
    "pam_kernel_test.py", # its parity harness: raw-torch oracle vs kernel
    "selftest.py",        # compares against plain F.cross_entropy, which speaks raw torch
    "train.py",           # optimiser / dataloader plumbing
    "generate.py",        # sampling loop
    "check_torch_layout.py",
})

# Functions that wrap a kernel sempyt cannot name, or that are the module's
# raw-torch public edge.
KERNEL_FUNCTIONS: dict[str, frozenset[str]] = {
    "complex_ops.py": frozenset({
        "build_rope_cache",     # position table built once, outside the graph
    }),
    "real_ops.py": frozenset({
        "build_rope_cache_real",  # real RoPE position table, built once
    }),
    "model.py": frozenset({
        "generate",                # sampling loop over raw logits
        "_stable_notebook",        # log-space decay-matrix scan: raw fp32, bounded backward
        "_chunked",                # real arm: hand-off to the fused PAM scan (triton_kernels)
        "_chunked_fused",          # complex arm: hand-off to fused_complex_pam_read
        "_short_conv",             # A1: depthwise causal Conv1d over the time axis
        "_chunked_multi",          # A2: multi-state batching into the kernel B*H*S axis
        "_chunked_delta",          # A3: delta erase/write hand-off to pam_delta_torch
    }),
}

# Contexts where raw torch is not an escape from named tensors: declaring
# parameters and buffers, type annotations, imports, and compile/grad modes.
ALLOW_CONTEXT = re.compile(
    r"^\s*(import|from|@)"
    r"|nn\.Parameter\("
    r"|register_buffer"
    r"|torch\.(no_grad|compile|Tensor|bool|long|float32)"
    r"|[:>]\s*torch\.Tensor"
    r"|torch\.(tensor|zeros|ones|linspace|arange)\("  # plain constructors
)

LAYOUT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.transpose\s*\("), "transpose"),
    (re.compile(r"\.unsqueeze\s*\("), "unsqueeze"),
    (re.compile(r"\.squeeze\s*\("), "squeeze"),
    (re.compile(r"\.permute\s*\("), "permute"),
    (re.compile(r"\.view\s*\("), "view"),
    (re.compile(r"\.reshape\s*\("), "reshape"),
    (re.compile(r"\.contiguous\s*\("), "contiguous"),
    (re.compile(r"\.expand\s*\("), "expand"),
    (re.compile(r"\[\.\.\.,"), "ellipsis index"),
    # A single-digit or negative dim= is a positional axis; `dim=384` is a width.
    (re.compile(r"\bdim\s*=\s*-?\d\b"), "numeric dim="),
]

ESCAPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.raw\s*\("), ".raw()"),
    (re.compile(r"\.data\b"), ".data"),
    (re.compile(r"\btorch\.\w"), "torch.*"),
]

_DEF_RE = re.compile(r"^\s*def (\w+)\(")
_ALLOW_RE = re.compile(r"#\s*named-exit:")
_FENCE_RE = re.compile(r'"""|\'\'\'')


def scan_file(path: Path) -> list[str]:
    if path.name in SKIP_FILES:
        return []
    allowed = KERNEL_FUNCTIONS.get(path.name, frozenset())
    current: str | None = None
    in_docstring = False
    hits: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fences = len(_FENCE_RE.findall(line))
        was_docstring = in_docstring
        if fences % 2:
            in_docstring = not in_docstring
        if was_docstring or fences:
            continue
        match = _DEF_RE.match(line)
        if match:
            current = match.group(1)
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or _ALLOW_RE.search(line):
            continue
        if current in allowed:
            continue
        checks = [(LAYOUT_PATTERNS, "layout")]
        if not ALLOW_CONTEXT.search(line):
            checks.append((ESCAPE_PATTERNS, "escape"))
        for patterns, kind in checks:
            found = next((label for pattern, label in patterns if pattern.search(line)), None)
            if found:
                hits.append(f"{path.relative_to(ROOT)}:{line_no}: {kind} {found}: {stripped}")
                break
    return hits


def main() -> int:
    # tmp/ holds throwaway benches and parity scripts (AGENTS: no commits, not
    # production); they speak raw torch on purpose.
    hits = [h for path in sorted(ROOT.glob("**/*.py"))
            if "tmp" not in path.parts for h in scan_file(path)]
    if hits:
        print("v13_sempty is reaching around sempyt:\n")
        for hit in hits:
            print(f"  {hit}")
        print(
            f"\n{len(hits)} violation(s). Use the named form, move the code into a "
            "declared kernel function, or mark the line '# named-exit: <why>'."
        )
        return 1
    print("OK — v13_sempty is named end to end outside declared boundaries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
