"""Timestamped logging helpers for duplex training / inference scripts."""

from __future__ import annotations

import time
from datetime import datetime


def ts() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(msg: str, *, flush: bool = True) -> None:
    print(f'[{ts()}] {msg}', flush=flush)


def elapsed_since(t0: float) -> str:
    s = max(0.0, time.time() - t0)
    if s < 60:
        return f'{s:.1f}s'
    if s < 3600:
        return f'{s / 60:.1f}m'
    return f'{s / 3600:.2f}h'
