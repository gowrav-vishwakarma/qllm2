"""Build the per-file eligible-doc index that makes old stream cursors instant.

A checkpoint's data cursor counts eligible (post-filter) docs per source. To
resume or warm-start without re-reading that prefix, ``v7.data._parquet_stream``
needs to know how many eligible docs each parquet file holds. This reads each
source in stream order until the file containing ``--until`` docs has been
read to the end; the reader itself records each completed file in
``.cache/v7_tokens/stream_index/``. ``--commit-copy`` then copies the index
into ``v7/stream_index/`` (tracked) so the other box gets it with ``git pull``.

  .venv/bin/python scripts/build_stream_index.py --source dclm --until 1184653
  .venv/bin/python scripts/build_stream_index.py --commit-copy

Uses the v13_sempty pretrain-mix filters (edu>=3, 5 % holdout excluded).
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from v7 import data  # noqa: E402


def build(source: str, until: int) -> None:
    spec = data.SOURCE_REGISTRY[source]
    counters: dict = {}
    it = data._open_source_iter(
        source, spec, edu_score_min=3, fineweb_name='sample-10BT',
        exclude_holdout=True, holdout_pct=5, skip=0, doc_counters=counters,
        mix_seed=42)
    fkey = f'{source}@f'
    t0, last_file = time.time(), 0
    for _ in it:
        f = counters.get(fkey, 0)
        if f != last_file:
            done = counters[source] - 1  # eligible docs in files [0, f)
            print(f'  [{source}] file {last_file} done: {done:,} docs in files < {f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
            last_file = f
            if done >= until:
                print(f'  [{source}] covers {until:,} -> stop', flush=True)
                return
    print(f'  [{source}] stream ended at {counters.get(source, 0):,} docs', flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--source', help='registry source name (dclm, fineweb, smoltalk2_mid, pg19)')
    p.add_argument('--until', type=int, default=0, help='eligible-doc cursor to cover')
    p.add_argument('--commit-copy', action='store_true',
                   help='copy .cache index files into v7/stream_index/ (tracked)')
    args = p.parse_args()
    if args.source:
        build(args.source, args.until)
    if args.commit_copy:
        src, dst = data._STREAM_INDEX_CACHE_DIR, data._STREAM_INDEX_REPO_DIR
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob('*.json')):
            shutil.copy2(f, dst / f.name)
            print(f'  copied {f.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
