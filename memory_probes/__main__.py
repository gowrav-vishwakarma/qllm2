"""Memory probes — evaluation framework for recurrent matrix memory.

Run:
    .venv/bin/python -m memory_probes --all
    .venv/bin/python -m memory_probes --test binding
    .venv/bin/python -m memory_probes --test language-filler
"""

from memory_probes.cli import main

__all__ = ['main']

if __name__ == '__main__':
    raise SystemExit(main())
