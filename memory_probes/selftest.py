"""Run implementation correctness checks (train ≡ recurrent)."""

from __future__ import annotations

import sys


def run_selftest() -> int:
    from v11.selftest import main
    return main()


if __name__ == '__main__':
    raise SystemExit(run_selftest())
