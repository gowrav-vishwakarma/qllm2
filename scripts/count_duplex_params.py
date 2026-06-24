#!/usr/bin/env python3
"""Print parameter breakdown for duplex presets."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v11.duplex.config import DUPLEX_PRESETS
from v11.duplex.model import V11DuplexLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', default='', help='Single preset or empty for all')
    args = p.parse_args()
    names = [args.preset] if args.preset else sorted(DUPLEX_PRESETS.keys())
    for name in names:
        cfg = DUPLEX_PRESETS[name]
        m = V11DuplexLM(cfg)
        counts = m.count_parameters()
        print(f"\n{name}  (dim={cfg.dim} L={cfg.n_layers} H={cfg.n_heads} K={cfg.n_states} V={cfg.vocab_size})")
        for k, v in counts.items():
            print(f"  {k}: {v:,} ({v/1e6:.3f}M)")
        print(f"  n_states={cfg.n_states} decay={cfg.decay_mode} write={cfg.write_mode}")


if __name__ == '__main__':
    main()
