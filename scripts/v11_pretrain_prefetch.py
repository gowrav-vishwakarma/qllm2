#!/usr/bin/env python3
"""Build on-disk pretrain token cache for the next round (CPU, resumable).

Eliminates the slow doc-skip + live tokenization at pretrain startup when the
cache already exists. Typical workflow is driven by ``run_v11_round.sh prefetch``;
this module is also callable directly.

Parallel prefetch while round N trains on GPU:
  - checkpoint = round (N-1) end cursors (start of round N)
  - offset_tokens = TOKEN_BUDGET (skip the 2B round N is consuming)
  - token_budget = TOKEN_BUDGET (cache round N+1's 2B slice)

After round N finishes:
  - checkpoint = round N end ``best_model.pt``
  - offset_tokens = 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Repo root on path when invoked as script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v7.data import build_pretrain_mix_token_cache, pretrain_mix_cache_dir, _pretrain_cache_meta


def _parse_sources_weights(sources: str, weights: str | None):
    src = tuple(x.strip() for x in sources.split(',') if x.strip())
    if weights:
        w = tuple(float(x.strip()) for x in weights.split(',') if x.strip())
        if len(w) != len(src):
            raise ValueError(f'weights ({len(w)}) must match sources ({len(src)})')
        return src, w
    return src, None


def _cursors_from_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return dict(ckpt.get('per_source_docs') or {})


def main() -> None:
    p = argparse.ArgumentParser(description='Prefetch pretrain mix token cache')
    p.add_argument('--checkpoint', required=True, help='Cursor source (.pt with per_source_docs)')
    p.add_argument('--token_budget', type=int, default=2_000_000_000)
    p.add_argument('--offset_tokens', type=int, default=0,
                   help='Skip first N training tokens before writing (parallel prefetch)')
    p.add_argument('--seq_len', type=int, default=2048)
    p.add_argument('--edu_score_min', type=int, default=3)
    p.add_argument('--pretrain_sources', default='dclm,fineweb,smoltalk2_mid')
    p.add_argument('--pretrain_weights', default='48,48,4')
    p.add_argument('--fineweb_name', default='sample-10BT')
    p.add_argument('--blend_warmup_tokens', type=int, default=0)
    p.add_argument('--mix_seed', type=int, default=42)
    p.add_argument('--no_resume', action='store_true')
    p.add_argument('--dry', action='store_true', help='Print resolved cache path only')
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f'checkpoint not found: {ckpt}', file=sys.stderr)
        sys.exit(1)

    sources, weights = _parse_sources_weights(args.pretrain_sources, args.pretrain_weights)
    docs = _cursors_from_checkpoint(ckpt)
    skip_map = {s: int(docs.get(s, 0)) for s in sources}

    meta = _pretrain_cache_meta(
        seq_len=args.seq_len,
        token_budget=args.token_budget,
        offset_tokens=args.offset_tokens,
        edu_score_min=args.edu_score_min,
        sources=sources,
        weights=weights,
        fineweb_name=args.fineweb_name,
        mix_seed=args.mix_seed,
        blend_warmup_tokens=args.blend_warmup_tokens,
        skip_map=skip_map,
    )
    cache_dir = pretrain_mix_cache_dir(meta)

    print(f'checkpoint : {ckpt}')
    print(f'start docs : {skip_map}')
    print(f'offset tok : {args.offset_tokens:,}')
    print(f'budget tok : {args.token_budget:,}')
    print(f'cache dir  : {cache_dir}')

    if args.dry:
        complete = (cache_dir / 'manifest.json').exists()
        print(f'status     : {"complete" if complete else "missing"}')
        return

    build_pretrain_mix_token_cache(
        seq_len=args.seq_len,
        token_budget=args.token_budget,
        offset_tokens=args.offset_tokens,
        edu_score_min=args.edu_score_min,
        sources=sources,
        weights=weights,
        fineweb_name=args.fineweb_name,
        mix_seed=args.mix_seed,
        blend_warmup_tokens=args.blend_warmup_tokens,
        skip_map=skip_map,
        resume=not args.no_resume,
    )


if __name__ == '__main__':
    main()
