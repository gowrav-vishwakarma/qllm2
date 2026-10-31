"""Training text pool for the library key head (L1): GPT-2 token ids from the last FineWeb
sample-10BT shard, never from the held-out prose the benchmark scores on.

    .venv/bin/python -m v14.train_pool --out .cache/v14_train_pool_fineweb014.pt --tokens 6000000

Format = the held-out cache's (dict with 'tokens'), so ``v16/bench.py --filler`` and
``memory_probes.spread.filler_tokens`` read it. Documents are joined by the GPT-2 EOS id
(50256), which ``filler_tokens`` drops, so the filler reads as continuous prose.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

SHARD = 'datasets/HuggingFaceFW/fineweb/sample/10BT/014_00000.parquet'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--tokens', type=int, default=6_000_000)
    ap.add_argument('--min_chars', type=int, default=2000, help='keep long documents only')
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained('gpt2')
    fs = HfFileSystem()
    t0, ids, docs = time.time(), [], 0
    with fs.open(SHARD, 'rb') as f:
        pf = pq.ParquetFile(f)
        print(f"{SHARD}: {pf.metadata.num_rows:,} docs in {pf.num_row_groups} row groups", flush=True)
        for g in range(pf.num_row_groups):
            texts = pf.read_row_group(g, columns=['text']).column('text').to_pylist()
            for t in texts:
                if len(t) < args.min_chars:
                    continue
                ids.extend(tok.encode(t) + [50256])
                docs += 1
            print(f"  row group {g}: {len(ids):,} tokens, {docs:,} docs, {time.time() - t0:.0f}s", flush=True)
            if len(ids) >= args.tokens:
                break
    arr = np.asarray(ids[:args.tokens], dtype=np.int64)
    torch.save({'tokens': torch.from_numpy(arr), 'source': SHARD, 'docs': docs}, args.out)
    print(f"saved {args.out}: {len(arr):,} tokens", flush=True)


if __name__ == '__main__':
    main()
