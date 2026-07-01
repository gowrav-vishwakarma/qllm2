#!/usr/bin/env python3
"""Compute deterministic data cursors for the next pretrain round.

Reads ``per_source_docs`` from a checkpoint (saved by the trainer) and emits the
skip cursors so the next ``+2B`` round consumes only fresh docs/rows -- no reuse.
Also suggests FineWeb config rotation when the current shard is near-exhausted.

The trainer also auto-seeds skips from ``--resume`` checkpoints, so this helper is
mainly for: starting a new round from ``best_model.pt`` with explicit values,
inspection, and driving ``scripts/run_v11_round.sh``.

Usage:
    # Human-readable
    uv run python scripts/v11_data_cursor.py \
        --checkpoint checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt

    # Shell-eval env vars (for the round runner)
    eval "$(uv run python scripts/v11_data_cursor.py --checkpoint ... --emit env)"
    #  -> DCLM_SKIP_DOCS=... FINEWEB_SKIP_DOCS=... SMOLTALK2_MID_SKIP_ROWS=... FINEWEB_NAME=...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Approx docs available per FineWeb config before rotating to the next shard.
# sample-10BT holds ~10B tokens; rotate to sample-100BT well before exhaustion.
FINEWEB_ROTATE = {
    'sample-10BT': ('sample-100BT', 9_000_000),   # ~docs; rotate past this cursor
    'sample-100BT': ('sample-350BT', 90_000_000),
}


def main() -> None:
    p = argparse.ArgumentParser(description='Compute next-round data cursors')
    p.add_argument('--checkpoint', required=True, help='Prev round best/latest .pt')
    p.add_argument('--fineweb-name', default='sample-10BT',
                   help='Current FineWeb config (for rotation suggestion)')
    p.add_argument('--emit', choices=['human', 'env'], default='human')
    args = p.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        print(f"# checkpoint not found: {path} (treating as fresh scratch: all skips 0)",
              file=sys.stderr)
        docs = {}
    else:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        docs = dict(ckpt.get('per_source_docs') or {})
        toks = dict(ckpt.get('per_source_tokens') or {})

    dclm = int(docs.get('dclm', 0))
    fineweb = int(docs.get('fineweb', 0))
    mid = int(docs.get('smoltalk2_mid', 0))

    fineweb_name = args.fineweb_name
    rotate = FINEWEB_ROTATE.get(fineweb_name)
    if rotate and fineweb >= rotate[1]:
        # Shard near-exhausted: rotate to the next config and reset its cursor.
        fineweb_name = rotate[0]
        fineweb = 0

    if args.emit == 'env':
        print(
            f"DCLM_SKIP_DOCS={dclm} "
            f"FINEWEB_SKIP_DOCS={fineweb} "
            f"SMOLTALK2_MID_SKIP_ROWS={mid} "
            f"FINEWEB_NAME={fineweb_name}"
        )
        return

    print(f"checkpoint: {path}")
    print(f"consumed docs/rows: {docs or '(none - scratch)'}")
    if path.exists():
        print(f"consumed tokens:    { {k: f'{v:,}' for k, v in toks.items()} }")
    print("next-round cursors:")
    print(f"  --dclm_skip_docs {dclm}")
    print(f"  --fineweb_skip_docs {fineweb}")
    print(f"  --smoltalk2_mid_skip_rows {mid}")
    print(f"  --fineweb_name {fineweb_name}")
    if fineweb_name != args.fineweb_name:
        print(f"  NOTE: rotated FineWeb {args.fineweb_name} -> {fineweb_name} (shard near-exhausted)")


if __name__ == '__main__':
    main()
