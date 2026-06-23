"""Deprecated: use `python -m memory_probes` instead.

This module re-exports the memory probes CLI for backward compatibility.
"""

from __future__ import annotations

import warnings

from memory_probes.cli import main

warnings.warn(
    'v11.pam_math is deprecated; use `python -m memory_probes`',
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['main']

if __name__ == '__main__':
    raise SystemExit(main())
